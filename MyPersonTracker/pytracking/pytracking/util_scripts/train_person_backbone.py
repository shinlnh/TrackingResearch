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
from torch.utils.data import DataLoader, Dataset

from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


def read_sequence_split(path: Path, prefix: str) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip().startswith(prefix)]


def load_boxes(seq_dir: Path) -> np.ndarray:
    return np.loadtxt(seq_dir / 'groundtruth.txt', delimiter=',', dtype=np.float32)


def load_visible(seq_dir: Path) -> np.ndarray:
    with open(seq_dir / 'full_occlusion.txt', newline='') as fh:
        occ = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    with open(seq_dir / 'out_of_view.txt', newline='') as fh:
        oov = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    return np.logical_not(np.logical_or(occ, oov))


@dataclass
class CropRecord:
    image_path: str
    bbox: tuple[float, float, float, float]
    label: int


def build_sequence_dir(root: Path, seq_name: str) -> Path:
    class_name = seq_name.split('-')[0]
    return root / class_name / seq_name


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


def sample_object_records(
    lasot_root: Path,
    seq_names: list[str],
    label: int,
    frame_stride: int,
    max_per_sequence: int,
    seed: int,
) -> list[CropRecord]:
    rng = random.Random(seed)
    records: list[CropRecord] = []
    for seq_name in seq_names:
        seq_dir = build_sequence_dir(lasot_root, seq_name)
        boxes = load_boxes(seq_dir)
        visible = load_visible(seq_dir)
        valid_ids = [idx for idx, is_visible in enumerate(visible) if is_visible and boxes[idx, 2] > 2 and boxes[idx, 3] > 2]
        valid_ids = valid_ids[::frame_stride]
        if len(valid_ids) > max_per_sequence:
            valid_ids = sorted(rng.sample(valid_ids, max_per_sequence))

        for frame_id in valid_ids:
            records.append(
                CropRecord(
                    image_path=str(seq_dir / 'img' / f'{frame_id + 1:08d}.jpg'),
                    bbox=tuple(float(v) for v in boxes[frame_id]),
                    label=label,
                )
            )
    return records


def sample_background_records(
    lasot_root: Path,
    seq_names: list[str],
    frame_stride: int,
    max_per_sequence: int,
    seed: int,
) -> list[CropRecord]:
    rng = random.Random(seed)
    records: list[CropRecord] = []
    for seq_name in seq_names:
        seq_dir = build_sequence_dir(lasot_root, seq_name)
        boxes = load_boxes(seq_dir)
        visible = load_visible(seq_dir)
        valid_ids = [idx for idx, is_visible in enumerate(visible) if is_visible and boxes[idx, 2] > 2 and boxes[idx, 3] > 2]
        valid_ids = valid_ids[::frame_stride]
        if len(valid_ids) > max_per_sequence:
            valid_ids = sorted(rng.sample(valid_ids, max_per_sequence))

        for frame_id in valid_ids:
            x, y, w, h = [float(v) for v in boxes[frame_id]]
            shift_x = rng.choice([-1.8, -1.4, 1.4, 1.8]) * w
            shift_y = rng.choice([-1.8, -1.4, 1.4, 1.8]) * h
            records.append(
                CropRecord(
                    image_path=str(seq_dir / 'img' / f'{frame_id + 1:08d}.jpg'),
                    bbox=(x + shift_x, y + shift_y, w, h),
                    label=0,
                )
            )
    return records


class BinaryCropDataset(Dataset):
    def __init__(self, records: list[CropRecord], image_size: int = 224, augment: bool = False):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = self._crop(image, record.bbox)

        if self.augment:
            crop = self._augment(crop)

        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor, torch.tensor(record.label, dtype=torch.long)

    def _crop(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        x, y, w, h = bbox
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        side = max(w, h) * 2.0
        x1, y1, x2, y2 = clamp_box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, image.shape[1], image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        if random.random() < 0.8:
            alpha = 0.9 + random.random() * 0.2
            beta = random.uniform(-12.0, 12.0)
            crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return crop


class PersonBackboneClassifier(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        self.backbone = resnet18_vggmconv1(['vggconv1', 'layer3'], path=weight_path)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(96 + 256, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(192, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        shallow = self.pool(features['vggconv1']).flatten(1)
        deep = self.pool(features['layer3']).flatten(1)
        return self.head(torch.cat([shallow, deep], dim=1))


def freeze_for_finetune(model: PersonBackboneClassifier):
    for param in model.backbone.parameters():
        param.requires_grad = False

    for module_name in ('vggmconv1', 'layer2', 'layer3'):
        module = getattr(model.backbone, module_name)
        for param in module.parameters():
            param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return {
        'loss': total_loss / max(total_samples, 1),
        'acc': total_correct / max(total_samples, 1),
        'samples': total_samples,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lasot_root = Path(args.lasot_root)
    split_root = lasot_root
    train_person = read_sequence_split(split_root / 'training_set.txt', 'person-')
    test_person = read_sequence_split(split_root / 'testing_set.txt', 'person-')
    train_non_person = [line.strip() for line in (split_root / 'training_set.txt').read_text().splitlines() if line.strip() and not line.startswith('person-')]
    test_non_person = [line.strip() for line in (split_root / 'testing_set.txt').read_text().splitlines() if line.strip() and not line.startswith('person-')]

    rng = random.Random(args.seed)
    rng.shuffle(train_non_person)
    rng.shuffle(test_non_person)
    train_non_person = train_non_person[: max(len(train_person) * 4, 32)]
    test_non_person = test_non_person[: max(len(test_person) * 4, 16)]

    train_records = []
    train_records.extend(sample_object_records(lasot_root, train_person, 1, args.person_stride, args.max_person_per_seq, args.seed))
    train_records.extend(sample_object_records(lasot_root, train_non_person, 0, args.other_stride, args.max_other_per_seq, args.seed + 1))
    train_records.extend(sample_background_records(lasot_root, train_person, args.person_stride, args.max_person_per_seq // 2, args.seed + 2))

    val_records = []
    val_records.extend(sample_object_records(lasot_root, test_person, 1, max(1, args.person_stride // 2), args.max_person_per_seq, args.seed + 3))
    val_records.extend(sample_object_records(lasot_root, test_non_person, 0, max(1, args.other_stride // 2), args.max_other_per_seq, args.seed + 4))
    val_records.extend(sample_background_records(lasot_root, test_person, max(1, args.person_stride // 2), max(1, args.max_person_per_seq // 2), args.seed + 5))

    train_dataset = BinaryCropDataset(train_records, image_size=args.image_size, augment=True)
    val_dataset = BinaryCropDataset(val_records, image_size=args.image_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
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
    model = PersonBackboneClassifier(args.base_weight).to(device)
    freeze_for_finetune(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / 'person_backbone_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_records': len(train_records),
        'val_records': len(val_records),
        'epochs': [],
    }
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_samples += int(labels.size(0))

        train_metrics = {
            'loss': running_loss / max(running_samples, 1),
            'acc': running_correct / max(running_samples, 1),
            'samples': running_samples,
        }
        val_metrics = evaluate(model, val_loader, device)

        history['epochs'].append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        torch.save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            },
            checkpoint_dir / f'epoch_{epoch:02d}.pth',
        )

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_state = {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

    if best_state is None:
        raise RuntimeError('Training did not produce a valid backbone state.')

    torch.save(best_state, output_path)
    history['best_val_acc'] = best_val_acc
    history['saved_weight'] = str(output_path)
    history['base_weight'] = args.base_weight
    history['device'] = str(device)

    log_path = output_path.with_suffix('.trainlog.json')
    log_path.write_text(json.dumps(history, indent=2))
    print(f'saved_backbone={output_path}')
    print(f'train_log={log_path}')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the ECO backbone for person tracking.')
    parser.add_argument('--lasot-root', type=str, required=True)
    parser.add_argument('--base-weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--person-stride', type=int, default=6)
    parser.add_argument('--other-stride', type=int, default=18)
    parser.add_argument('--max-person-per-seq', type=int, default=48)
    parser.add_argument('--max-other-per-seq', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
