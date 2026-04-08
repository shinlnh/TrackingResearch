import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


def read_person_split(path: Path, prefix: str = 'person-') -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip().startswith(prefix)]


def build_sequence_dir(root: Path, seq_name: str) -> Path:
    class_name = seq_name.split('-')[0]
    return root / class_name / seq_name


def load_boxes(seq_dir: Path) -> np.ndarray:
    return np.loadtxt(seq_dir / 'groundtruth.txt', delimiter=',', dtype=np.float32)


def load_visible(seq_dir: Path) -> np.ndarray:
    with open(seq_dir / 'full_occlusion.txt', newline='') as fh:
        occ = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    with open(seq_dir / 'out_of_view.txt', newline='') as fh:
        oov = np.array([int(v) for v in next(csv.reader(fh))], dtype=np.uint8)
    return np.logical_not(np.logical_or(occ, oov))


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


def crop_person(image: np.ndarray, bbox: np.ndarray, image_size: int, context_factor: float) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    side = max(w, h) * context_factor
    x1, y1, x2, y2 = clamp_box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, image.shape[1], image.shape[0])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return crop


def compute_projection(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lasot_root = Path(args.lasot_root)
    train_person = read_person_split(lasot_root / 'training_set.txt')

    sampled = []
    for seq_name in train_person:
        seq_dir = build_sequence_dir(lasot_root, seq_name)
        boxes = load_boxes(seq_dir)
        visible = load_visible(seq_dir)
        valid_ids = [idx for idx, is_visible in enumerate(visible) if is_visible and boxes[idx, 2] > 4 and boxes[idx, 3] > 4]
        valid_ids = valid_ids[::max(1, args.frame_stride)]
        if len(valid_ids) > args.max_per_sequence:
            valid_ids = sorted(random.sample(valid_ids, args.max_per_sequence))
        for frame_id in valid_ids:
            sampled.append((seq_dir / 'img' / f'{frame_id + 1:08d}.jpg', boxes[frame_id]))

    if not sampled:
        raise RuntimeError('No person crops found for projection estimation.')

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = resnet18_vggmconv1(['vggconv1', 'layer3'], path=args.base_weight).to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    shallow_chunks = []
    deep_chunks = []
    with torch.inference_mode():
        for batch_start in range(0, len(sampled), args.batch_size):
            batch = sampled[batch_start:batch_start + args.batch_size]
            crops = []
            for image_path, bbox in batch:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                crop = crop_person(image, bbox, args.image_size, args.context_factor)
                crops.append(crop.transpose(2, 0, 1))

            if not crops:
                continue

            tensor = torch.from_numpy(np.stack(crops)).float().to(device) / 255.0
            tensor = (tensor - mean) / std
            features = model(tensor)

            shallow = features['vggconv1'].permute(1, 0, 2, 3).reshape(features['vggconv1'].shape[1], -1)
            deep = features['layer3'].permute(1, 0, 2, 3).reshape(features['layer3'].shape[1], -1)
            shallow_chunks.append(shallow.float().cpu())
            deep_chunks.append(deep.float().cpu())

    if not shallow_chunks or not deep_chunks:
        raise RuntimeError('Projection extraction produced no features.')

    shallow_mat = torch.cat(shallow_chunks, dim=1)
    deep_mat = torch.cat(deep_chunks, dim=1)
    shallow_mat -= shallow_mat.mean(dim=1, keepdim=True)
    deep_mat -= deep_mat.mean(dim=1, keepdim=True)

    shallow_u = torch.linalg.svd(shallow_mat @ shallow_mat.t(), full_matrices=False).U[:, :args.shallow_dim].contiguous()
    deep_u = torch.linalg.svd(deep_mat @ deep_mat.t(), full_matrices=False).U[:, :args.deep_dim].contiguous()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'projection_matrix': [shallow_u, deep_u],
            'meta': {
                'base_weight': args.base_weight,
                'samples': len(sampled),
                'image_size': args.image_size,
                'context_factor': args.context_factor,
                'frame_stride': args.frame_stride,
                'max_per_sequence': args.max_per_sequence,
            },
        },
        output_path,
    )
    print(f'saved_projection={output_path}')
    print(f'samples={len(sampled)}')
    print(f'shallow_shape={tuple(shallow_u.shape)} deep_shape={tuple(deep_u.shape)}')


def main():
    parser = argparse.ArgumentParser(description='Build a person-specific ECO projection matrix from LaSOT person.')
    parser.add_argument('--lasot-root', type=str, required=True)
    parser.add_argument('--base-weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--context-factor', type=float, default=2.2)
    parser.add_argument('--frame-stride', type=int, default=8)
    parser.add_argument('--max-per-sequence', type=int, default=48)
    parser.add_argument('--shallow-dim', type=int, default=16)
    parser.add_argument('--deep-dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    compute_projection(args)


if __name__ == '__main__':
    main()
