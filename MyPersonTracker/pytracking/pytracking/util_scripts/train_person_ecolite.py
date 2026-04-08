import argparse
import csv
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

from ltr.models.backbone.person_ecolite import person_ecolite
from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


def read_lasot_person_split(root: Path, split_file: str) -> list[str]:
    return [line.strip() for line in (root / split_file).read_text(encoding='utf-8').splitlines() if line.strip().startswith('person-')]


def read_got10k_person_sequences(root: Path) -> list[str]:
    seq_names = [line.strip() for line in (root / 'list.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    person_like = []
    for seq_name in seq_names:
        meta_path = root / seq_name / 'meta_info.ini'
        text = meta_path.read_text(encoding='utf-8', errors='ignore').lower()
        if 'object_class: person' in text or 'object_class: human' in text or 'major_class: person' in text or 'major_class: human' in text:
            person_like.append(seq_name)
    return person_like


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


@dataclass
class SequenceMeta:
    source: str
    name: str
    frames_dir: Path
    boxes: np.ndarray
    visible_ids: list[int]


def load_lasot_sequence(root: Path, seq_name: str) -> SequenceMeta | None:
    class_name = seq_name.split('-')[0]
    seq_dir = root / class_name / seq_name
    boxes = np.loadtxt(seq_dir / 'groundtruth.txt', delimiter=',', dtype=np.float32)
    with open(seq_dir / 'full_occlusion.txt', newline='') as fh:
        occ = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    with open(seq_dir / 'out_of_view.txt', newline='') as fh:
        oov = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    visible = np.logical_not(np.logical_or(occ, oov))
    visible_ids = [idx for idx, is_visible in enumerate(visible) if is_visible and boxes[idx, 2] > 4 and boxes[idx, 3] > 4]
    if len(visible_ids) < 2:
        return None
    return SequenceMeta('lasot', seq_name, seq_dir / 'img', boxes, visible_ids)


def load_got10k_sequence(root: Path, seq_name: str) -> SequenceMeta | None:
    seq_dir = root / seq_name
    boxes = np.loadtxt(seq_dir / 'groundtruth.txt', delimiter=',', dtype=np.float32)
    with open(seq_dir / 'absence.label', newline='') as fh:
        absence = np.array([int(v[0]) for v in csv.reader(fh)], dtype=np.uint8)
    with open(seq_dir / 'cover.label', newline='') as fh:
        cover = np.array([int(v[0]) for v in csv.reader(fh)], dtype=np.uint8)
    visible = np.logical_and(absence == 0, cover > 0)
    visible_ids = [idx for idx, is_visible in enumerate(visible) if is_visible and boxes[idx, 2] > 4 and boxes[idx, 3] > 4]
    if len(visible_ids) < 2:
        return None
    return SequenceMeta('got10k', seq_name, seq_dir, boxes, visible_ids)


def build_person_sequence_pool(lasot_root: Path, got10k_root: Path):
    lasot_train = []
    lasot_val = []
    for seq_name in read_lasot_person_split(lasot_root, 'training_set.txt'):
        seq = load_lasot_sequence(lasot_root, seq_name)
        if seq is not None:
            lasot_train.append(seq)
    for seq_name in read_lasot_person_split(lasot_root, 'testing_set.txt'):
        seq = load_lasot_sequence(lasot_root, seq_name)
        if seq is not None:
            lasot_val.append(seq)

    got_person = []
    for seq_name in read_got10k_person_sequences(got10k_root):
        seq = load_got10k_sequence(got10k_root, seq_name)
        if seq is not None:
            got_person.append(seq)

    random.shuffle(got_person)
    got_val_count = max(32, min(160, len(got_person) // 15))
    got_val = got_person[:got_val_count]
    got_train = got_person[got_val_count:]
    return lasot_train, lasot_val, got_train, got_val


class MixedPersonPairDataset(Dataset):
    def __init__(
        self,
        lasot_sequences: list[SequenceMeta],
        got_sequences: list[SequenceMeta],
        pairs_per_epoch: int,
        got_probability: float = 0.7,
        image_size: int = 224,
        context_factor: float = 2.2,
        jitter: float = 0.15,
        max_gap: int = 80,
        augment: bool = False,
        seed: int = 2026,
    ):
        self.lasot_sequences = lasot_sequences
        self.got_sequences = got_sequences
        self.pairs_per_epoch = pairs_per_epoch
        self.got_probability = got_probability
        self.image_size = image_size
        self.context_factor = context_factor
        self.jitter = jitter
        self.max_gap = max_gap
        self.augment = augment
        self.seed = seed
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        use_got = self.got_sequences and (not self.lasot_sequences or rng.random() < self.got_probability)
        pool = self.got_sequences if use_got else self.lasot_sequences
        seq = pool[rng.randrange(len(pool))]
        template_id, search_id = self._sample_pair(seq.visible_ids, rng)
        template = self._load_crop(seq, template_id, rng)
        search = self._load_crop(seq, search_id, rng)
        return template, search

    def _sample_pair(self, visible_ids: list[int], rng: random.Random) -> tuple[int, int]:
        template_id = rng.choice(visible_ids)
        candidates = [fid for fid in visible_ids if abs(fid - template_id) <= self.max_gap]
        if not candidates:
            candidates = visible_ids
        search_id = rng.choice(candidates)
        return template_id, search_id

    def _frame_path(self, seq: SequenceMeta, frame_id: int) -> Path:
        return seq.frames_dir / f'{frame_id + 1:08d}.jpg'

    def _load_crop(self, seq: SequenceMeta, frame_id: int, rng: random.Random) -> torch.Tensor:
        image_path = self._frame_path(seq, frame_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x, y, w, h = [float(v) for v in seq.boxes[frame_id]]
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        side = max(w, h) * self.context_factor
        scale_jitter = 1.0 + rng.uniform(-self.jitter, self.jitter)
        shift_x = rng.uniform(-self.jitter, self.jitter) * w
        shift_y = rng.uniform(-self.jitter, self.jitter) * h
        side *= scale_jitter
        cx += shift_x
        cy += shift_y
        x1, y1, x2, y2 = clamp_box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, image.shape[1], image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            if rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1])
            if rng.random() < 0.8:
                alpha = 0.85 + rng.random() * 0.3
                beta = rng.uniform(-14.0, 14.0)
                crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if rng.random() < 0.3:
                sigma = rng.uniform(0.0, 1.0)
                crop = cv2.GaussianBlur(crop, (3, 3), sigmaX=sigma)

        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor


class PersonEcoLiteDistillModel(nn.Module):
    def __init__(self, teacher_weight_path: str):
        super().__init__()
        self.student = person_ecolite(['stem', 'stage4'])
        self.teacher = resnet18_vggmconv1(['vggconv1', 'layer3'], path=teacher_weight_path)
        self.student_to_teacher_shallow = nn.Conv2d(24, 96, kernel_size=1)
        self.student_to_teacher_deep = nn.Conv2d(96, 256, kernel_size=1)
        self.embedding_head = nn.Sequential(
            nn.Linear(24 + 96, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def encode_student(self, x: torch.Tensor):
        features = self.student(x)
        shallow = features['stem']
        deep = features['stage4']
        pooled = torch.cat([
            F.adaptive_avg_pool2d(shallow, 1).flatten(1),
            F.adaptive_avg_pool2d(deep, 1).flatten(1),
        ], dim=1)
        embedding = F.normalize(self.embedding_head(pooled), dim=1)
        return shallow, deep, embedding

    def encode_teacher(self, x: torch.Tensor):
        with torch.no_grad():
            features = self.teacher(x)
        return features['vggconv1'], features['layer3']


def contrastive_loss(template_embedding: torch.Tensor, search_embedding: torch.Tensor, temperature: float):
    logits = (template_embedding @ search_embedding.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    acc = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, acc


def normalized_mse(student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
    student_map = F.normalize(student_map.flatten(2), dim=1)
    teacher_map = F.normalize(teacher_map.flatten(2), dim=1)
    return F.mse_loss(student_map, teacher_map)


def build_losses(model: PersonEcoLiteDistillModel, template: torch.Tensor, search: torch.Tensor, temperature: float, distill_weight: float):
    t_shallow_s, t_deep_s, t_embed = model.encode_student(template)
    s_shallow_s, s_deep_s, s_embed = model.encode_student(search)

    t_shallow_t, t_deep_t = model.encode_teacher(template)
    s_shallow_t, s_deep_t = model.encode_teacher(search)

    c_loss, acc = contrastive_loss(t_embed, s_embed, temperature)

    shallow_loss = normalized_mse(model.student_to_teacher_shallow(t_shallow_s), t_shallow_t)
    shallow_loss = shallow_loss + normalized_mse(model.student_to_teacher_shallow(s_shallow_s), s_shallow_t)

    deep_loss = normalized_mse(model.student_to_teacher_deep(t_deep_s), t_deep_t)
    deep_loss = deep_loss + normalized_mse(model.student_to_teacher_deep(s_deep_s), s_deep_t)

    distill = 0.5 * shallow_loss + deep_loss
    total = c_loss + distill_weight * distill
    return total, c_loss, distill, acc


def evaluate(model: PersonEcoLiteDistillModel, loader: DataLoader, device: torch.device, temperature: float, distill_weight: float):
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for template, search in loader:
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            total, _, _, acc = build_losses(model, template, search, temperature, distill_weight)
            losses.append(float(total.item()))
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
    got10k_root = Path(args.got10k_root)
    lasot_train, lasot_val, got_train, got_val = build_person_sequence_pool(lasot_root, got10k_root)

    train_dataset = MixedPersonPairDataset(
        lasot_train,
        got_train,
        pairs_per_epoch=args.pairs_per_epoch,
        got_probability=args.got_probability,
        image_size=args.image_size,
        context_factor=args.context_factor,
        jitter=args.jitter,
        max_gap=args.max_gap,
        augment=True,
        seed=args.seed,
    )
    val_dataset = MixedPersonPairDataset(
        lasot_val,
        got_val,
        pairs_per_epoch=max(args.val_pairs, args.batch_size * 8),
        got_probability=0.5,
        image_size=args.image_size,
        context_factor=args.context_factor,
        jitter=max(0.05, args.jitter * 0.5),
        max_gap=args.max_gap,
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
    model = PersonEcoLiteDistillModel(args.teacher_weight).to(device)
    model.teacher.eval()
    for param in model.teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / 'person_ecolite_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'teacher_weight': args.teacher_weight,
        'device': str(device),
        'lasot_train_sequences': len(lasot_train),
        'lasot_val_sequences': len(lasot_val),
        'got10k_train_sequences': len(got_train),
        'got10k_val_sequences': len(got_val),
        'epochs': [],
    }
    best_score = -math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.teacher.eval()
        train_losses = []
        train_accs = []
        for template, search in train_loader:
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda' and args.use_amp)):
                total, _, _, acc = build_losses(model, template, search, args.temperature, args.distill_weight)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(total.item()))
            train_accs.append(acc)

        train_metrics = {
            'loss': float(np.mean(train_losses)) if train_losses else math.inf,
            'retrieval_acc': float(np.mean(train_accs)) if train_accs else 0.0,
            'steps': len(train_losses),
        }
        val_metrics = evaluate(model, val_loader, device, args.temperature, args.distill_weight)
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
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'score': score,
            },
            checkpoint_dir / f'epoch_{epoch:02d}.pth',
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.student.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['retrieval_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['retrieval_acc']:.4f} score={score:.4f}"
        )

    if best_state is None:
        raise RuntimeError('Training did not produce a valid lightweight backbone state.')

    torch.save(best_state, output_path)
    history['best_score'] = best_score
    history['saved_weight'] = str(output_path)
    log_path = output_path.with_suffix('.trainlog.json')
    log_path.write_text(json.dumps(history, indent=2))
    print(f'saved_backbone={output_path}')
    print(f'train_log={log_path}')


def main():
    parser = argparse.ArgumentParser(description='Train a lightweight ECO backbone for person tracking.')
    parser.add_argument('--lasot-root', type=str, required=True)
    parser.add_argument('--got10k-root', type=str, required=True)
    parser.add_argument('--teacher-weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--distill-weight', type=float, default=0.15)
    parser.add_argument('--pairs-per-epoch', type=int, default=2048)
    parser.add_argument('--val-pairs', type=int, default=384)
    parser.add_argument('--got-probability', type=float, default=0.7)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--context-factor', type=float, default=2.2)
    parser.add_argument('--jitter', type=float, default=0.15)
    parser.add_argument('--max-gap', type=int, default=80)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--use-amp', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
