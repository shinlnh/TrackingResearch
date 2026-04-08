import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


def read_split(path: Path, prefix: str) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip().startswith(prefix)]


def build_sequence_dir(root: Path, seq_name: str) -> Path:
    class_name = seq_name.split('-')[0]
    return root / class_name / seq_name


def load_boxes(seq_dir: Path) -> np.ndarray:
    return np.loadtxt(seq_dir / 'groundtruth.txt', delimiter=',', dtype=np.float32)


def load_visible(seq_dir: Path) -> np.ndarray:
    occ = np.loadtxt(seq_dir / 'full_occlusion.txt', delimiter=',', dtype=np.uint8).reshape(-1)
    oov = np.loadtxt(seq_dir / 'out_of_view.txt', delimiter=',', dtype=np.uint8).reshape(-1)
    return np.logical_not(np.logical_or(occ, oov))


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


@dataclass
class SequenceMeta:
    name: str
    frames_dir: Path
    boxes: np.ndarray
    visible_ids: list[int]


class PairCropDataset(Dataset):
    def __init__(
        self,
        sequence_metas: list[SequenceMeta],
        pairs_per_epoch: int,
        image_size: int = 224,
        max_gap: int = 80,
        context_factor: float = 2.2,
        jitter: float = 0.15,
        augment: bool = False,
        seed: int = 2026,
    ):
        self.sequence_metas = sequence_metas
        self.pairs_per_epoch = pairs_per_epoch
        self.image_size = image_size
        self.max_gap = max_gap
        self.context_factor = context_factor
        self.jitter = jitter
        self.augment = augment
        self.seed = seed
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        seq = self.sequence_metas[index % len(self.sequence_metas)]
        template_id, search_id = self._sample_pair(seq.visible_ids, rng)
        template = self._load_crop(seq, template_id, rng, self.augment)
        search = self._load_crop(seq, search_id, rng, self.augment)
        return template, search, torch.tensor(index % len(self.sequence_metas), dtype=torch.long)

    def _sample_pair(self, visible_ids: list[int], rng: random.Random) -> tuple[int, int]:
        template_id = rng.choice(visible_ids)
        candidates = [fid for fid in visible_ids if abs(fid - template_id) <= self.max_gap]
        if not candidates:
            candidates = visible_ids
        search_id = rng.choice(candidates)
        return template_id, search_id

    def _load_crop(self, seq: SequenceMeta, frame_id: int, rng: random.Random, augment: bool) -> torch.Tensor:
        image_path = seq.frames_dir / f'{frame_id + 1:08d}.jpg'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = self._crop(image, seq.boxes[frame_id], rng)
        if augment:
            crop = self._augment(crop, rng)
        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _crop(self, image: np.ndarray, box: np.ndarray, rng: random.Random) -> np.ndarray:
        x, y, w, h = [float(v) for v in box]
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        size = max(w, h) * self.context_factor
        jitter_scale = 1.0 + rng.uniform(-self.jitter, self.jitter)
        jitter_dx = rng.uniform(-self.jitter, self.jitter) * w
        jitter_dy = rng.uniform(-self.jitter, self.jitter) * h
        size *= jitter_scale
        cx += jitter_dx
        cy += jitter_dy
        x1, y1, x2, y2 = clamp_box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, image.shape[1], image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _augment(self, crop: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        if rng.random() < 0.8:
            alpha = 0.85 + rng.random() * 0.3
            beta = rng.uniform(-16.0, 16.0)
            crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            sigma = rng.uniform(0.0, 1.2)
            crop = cv2.GaussianBlur(crop, (3, 3), sigmaX=sigma)
        return crop


class PersonMetricBackbone(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        self.backbone = resnet18_vggmconv1(['vggconv1', 'layer3'], path=weight_path)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(96 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def encode(self, x: torch.Tensor):
        features = self.backbone(x)
        shallow = self.pool(features['vggconv1']).flatten(1)
        deep = self.pool(features['layer3']).flatten(1)
        concat = torch.cat([shallow, deep], dim=1)
        emb = F.normalize(self.proj(concat), dim=1)
        return emb, shallow, deep


def build_sequence_metas(lasot_root: Path, seq_names: list[str]) -> list[SequenceMeta]:
    metas = []
    for seq_name in seq_names:
        seq_dir = build_sequence_dir(lasot_root, seq_name)
        boxes = load_boxes(seq_dir)
        visible = load_visible(seq_dir)
        visible_ids = [
            idx for idx, is_visible in enumerate(visible)
            if is_visible and boxes[idx, 2] > 4 and boxes[idx, 3] > 4
        ]
        if len(visible_ids) < 2:
            continue
        metas.append(
            SequenceMeta(
                name=seq_name,
                frames_dir=seq_dir / 'img',
                boxes=boxes,
                visible_ids=visible_ids,
            )
        )
    if not metas:
        raise RuntimeError('No usable person sequences were found for metric training.')
    return metas


def freeze_for_metric_finetune(model: PersonMetricBackbone):
    for param in model.backbone.parameters():
        param.requires_grad = False

    for module_name in ('vggmconv1', 'layer2', 'layer3'):
        module = getattr(model.backbone, module_name)
        for param in module.parameters():
            param.requires_grad = True

    for param in model.proj.parameters():
        param.requires_grad = True


def contrastive_loss(template_emb: torch.Tensor, search_emb: torch.Tensor, temperature: float) -> tuple[torch.Tensor, float]:
    logits = template_emb @ search_emb.t()
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    acc = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, acc


def distill_loss(
    student_shallow: torch.Tensor,
    student_deep: torch.Tensor,
    teacher_shallow: torch.Tensor,
    teacher_deep: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(student_shallow, teacher_shallow) + F.mse_loss(student_deep, teacher_deep)


def evaluate(student: PersonMetricBackbone, teacher: PersonMetricBackbone, loader: DataLoader, device: torch.device, temperature: float) -> dict:
    student.eval()
    teacher.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for template, search, _ in loader:
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            s_temp_emb, s_temp_shallow, s_temp_deep = student.encode(template)
            s_search_emb, s_search_shallow, s_search_deep = student.encode(search)
            t_temp_emb, t_temp_shallow, t_temp_deep = teacher.encode(template)
            t_search_emb, t_search_shallow, t_search_deep = teacher.encode(search)
            c_loss, acc = contrastive_loss(s_temp_emb, s_search_emb, temperature)
            d_loss = (
                distill_loss(s_temp_shallow, s_temp_deep, t_temp_shallow, t_temp_deep) +
                distill_loss(s_search_shallow, s_search_deep, t_search_shallow, t_search_deep)
            )
            losses.append(float((c_loss + 0.1 * d_loss).item()))
            accs.append(acc)
    return {
        'loss': float(np.mean(losses)) if losses else math.inf,
        'retrieval_acc': float(np.mean(accs)) if accs else 0.0,
        'steps': len(losses),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    lasot_root = Path(args.lasot_root)
    train_person = read_split(lasot_root / 'training_set.txt', 'person-')
    test_person = read_split(lasot_root / 'testing_set.txt', 'person-')

    train_metas = build_sequence_metas(lasot_root, train_person)
    val_metas = build_sequence_metas(lasot_root, test_person)

    train_dataset = PairCropDataset(
        train_metas,
        pairs_per_epoch=args.pairs_per_epoch,
        image_size=args.image_size,
        max_gap=args.max_gap,
        context_factor=args.context_factor,
        jitter=args.jitter,
        augment=True,
        seed=args.seed,
    )
    val_dataset = PairCropDataset(
        val_metas,
        pairs_per_epoch=max(args.batch_size * 8, args.val_pairs),
        image_size=args.image_size,
        max_gap=args.max_gap,
        context_factor=args.context_factor,
        jitter=max(0.05, args.jitter * 0.5),
        augment=False,
        seed=args.seed + 1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    student = PersonMetricBackbone(args.base_weight).to(device)
    teacher = PersonMetricBackbone(args.base_weight).to(device)
    freeze_for_metric_finetune(student)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / 'person_metric_backbone_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'base_weight': args.base_weight,
        'device': str(device),
        'train_sequences': [m.name for m in train_metas],
        'val_sequences': [m.name for m in val_metas],
        'epochs': [],
    }
    best_score = -math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_losses = []
        epoch_accs = []
        for template, search, _ in train_loader:
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)

            s_temp_emb, s_temp_shallow, s_temp_deep = student.encode(template)
            s_search_emb, s_search_shallow, s_search_deep = student.encode(search)
            with torch.no_grad():
                _, t_temp_shallow, t_temp_deep = teacher.encode(template)
                _, t_search_shallow, t_search_deep = teacher.encode(search)

            c_loss, acc = contrastive_loss(s_temp_emb, s_search_emb, args.temperature)
            d_loss = (
                distill_loss(s_temp_shallow, s_temp_deep, t_temp_shallow, t_temp_deep) +
                distill_loss(s_search_shallow, s_search_deep, t_search_shallow, t_search_deep)
            )
            loss = c_loss + args.distill_weight * d_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_accs.append(acc)

        train_metrics = {
            'loss': float(np.mean(epoch_losses)) if epoch_losses else math.inf,
            'retrieval_acc': float(np.mean(epoch_accs)) if epoch_accs else 0.0,
            'steps': len(epoch_losses),
        }
        val_metrics = evaluate(student, teacher, val_loader, device, args.temperature)
        score = val_metrics['retrieval_acc'] - 0.1 * val_metrics['loss']

        history['epochs'].append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'score': score,
        })

        torch.save(
            {
                'epoch': epoch,
                'model_state': student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'score': score,
            },
            checkpoint_dir / f'epoch_{epoch:02d}.pth',
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in student.backbone.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['retrieval_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['retrieval_acc']:.4f} score={score:.4f}"
        )

    if best_state is None:
        raise RuntimeError('Metric training did not produce a valid backbone state.')

    torch.save(best_state, output_path)
    history['best_score'] = best_score
    history['saved_weight'] = str(output_path)
    log_path = output_path.with_suffix('.trainlog.json')
    log_path.write_text(json.dumps(history, indent=2))
    print(f'saved_backbone={output_path}')
    print(f'train_log={log_path}')


def main():
    parser = argparse.ArgumentParser(description='Tracking-aligned metric fine-tuning for the ECO person backbone.')
    parser.add_argument('--lasot-root', type=str, required=True)
    parser.add_argument('--base-weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--distill-weight', type=float, default=0.10)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--pairs-per-epoch', type=int, default=1024)
    parser.add_argument('--val-pairs', type=int, default=256)
    parser.add_argument('--max-gap', type=int, default=80)
    parser.add_argument('--context-factor', type=float, default=2.2)
    parser.add_argument('--jitter', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
