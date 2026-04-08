import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from ltr.models.backbone.person_ecolite import person_ecolite
from pytracking.util_scripts.train_person_ecolite import (
    load_got10k_sequence,
    load_lasot_sequence,
    read_got10k_person_sequences,
    read_lasot_person_split,
    clamp_box,
)


def crop_patch(image: np.ndarray, bbox: np.ndarray, image_size: int, context_factor: float) -> np.ndarray:
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


def collect_samples(lasot_root: Path, got10k_root: Path, lasot_max_per_seq: int, got_max_per_seq: int, frame_stride: int):
    samples = []
    for seq_name in read_lasot_person_split(lasot_root, 'training_set.txt'):
        seq = load_lasot_sequence(lasot_root, seq_name)
        if seq is None:
            continue
        visible_ids = seq.visible_ids[::max(1, frame_stride)][:lasot_max_per_seq]
        for frame_id in visible_ids:
            samples.append((seq.frames_dir / f'{frame_id + 1:08d}.jpg', seq.boxes[frame_id]))

    for seq_name in read_got10k_person_sequences(got10k_root):
        seq = load_got10k_sequence(got10k_root, seq_name)
        if seq is None:
            continue
        visible_ids = seq.visible_ids[::max(1, frame_stride)][:got_max_per_seq]
        for frame_id in visible_ids:
            samples.append((seq.frames_dir / f'{frame_id + 1:08d}.jpg', seq.boxes[frame_id]))
    return samples


def build_projection(args):
    lasot_root = Path(args.lasot_root)
    got10k_root = Path(args.got10k_root)
    samples = collect_samples(
        lasot_root,
        got10k_root,
        lasot_max_per_seq=args.lasot_max_per_seq,
        got_max_per_seq=args.got10k_max_per_seq,
        frame_stride=args.frame_stride,
    )
    if not samples:
        raise RuntimeError('No samples found for lightweight projection estimation.')

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = person_ecolite(['stem', 'stage4'], path=args.weight).to(device)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    shallow_chunks = []
    deep_chunks = []
    with torch.inference_mode():
        for batch_start in range(0, len(samples), args.batch_size):
            batch = samples[batch_start:batch_start + args.batch_size]
            crops = []
            for image_path, bbox in batch:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                crop = crop_patch(image, bbox, args.image_size, args.context_factor)
                crops.append(crop.transpose(2, 0, 1))
            if not crops:
                continue

            tensor = torch.from_numpy(np.stack(crops)).float().to(device) / 255.0
            tensor = (tensor - mean) / std
            features = model(tensor)
            shallow = features['stem'].permute(1, 0, 2, 3).reshape(features['stem'].shape[1], -1)
            deep = features['stage4'].permute(1, 0, 2, 3).reshape(features['stage4'].shape[1], -1)
            shallow_chunks.append(shallow.float().cpu())
            deep_chunks.append(deep.float().cpu())

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
                'weight': args.weight,
                'samples': len(samples),
                'lasot_root': str(lasot_root),
                'got10k_root': str(got10k_root),
            },
        },
        output_path,
    )
    print(f'saved_projection={output_path}')
    print(f'samples={len(samples)}')
    print(f'shallow_shape={tuple(shallow_u.shape)} deep_shape={tuple(deep_u.shape)}')


def main():
    parser = argparse.ArgumentParser(description='Build projection matrix for lightweight person ECO backbone.')
    parser.add_argument('--lasot-root', type=str, required=True)
    parser.add_argument('--got10k-root', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--context-factor', type=float, default=2.2)
    parser.add_argument('--frame-stride', type=int, default=8)
    parser.add_argument('--lasot-max-per-seq', type=int, default=48)
    parser.add_argument('--got10k-max-per-seq', type=int, default=8)
    parser.add_argument('--shallow-dim', type=int, default=8)
    parser.add_argument('--deep-dim', type=int, default=24)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    build_projection(args)


if __name__ == '__main__':
    main()
