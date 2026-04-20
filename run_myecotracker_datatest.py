#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MyECOTracker on the local datatest sequence and export simple diagnostics."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datatest") / "v01_normal_1",
        help="Path to the datatest sequence root.",
    )
    parser.add_argument("--tracker-name", default="eco", help="Tracker module name.")
    parser.add_argument(
        "--param",
        default="verified_otb936",
        help="Tracker parameter alias under pytracking.parameter.<tracker-name>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "myecotracker_datatest",
        help="Directory for metrics, predictions, and preview renders.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Export an MP4 overlay preview.",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=12,
        help="Number of evenly sampled overlay frames to export.",
    )
    return parser.parse_args()


def xywhn_to_xywh_pixels(label_parts: list[str], width: int, height: int) -> list[float]:
    if len(label_parts) != 6:
        raise ValueError(f"Expected 6 columns, got {len(label_parts)}: {label_parts}")

    _, x_center, y_center, box_w, box_h, _track_id = map(float, label_parts)
    x = (x_center - box_w / 2.0) * width
    y = (y_center - box_h / 2.0) * height
    return [x, y, box_w * width, box_h * height]


def load_sequence(dataset_root: Path, max_frames: int | None) -> tuple[list[str], np.ndarray]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    frames = sorted(image_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {image_dir}")

    if max_frames is not None:
        frames = frames[:max_frames]

    first_image = cv2.imread(str(frames[0]))
    if first_image is None:
        raise RuntimeError(f"Failed to read {frames[0]}")
    height, width = first_image.shape[:2]

    gt_rows: list[list[float]] = []
    for frame_path in frames:
        label_path = label_dir / f"{frame_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {frame_path.name}: {label_path}")
        label_parts = label_path.read_text(encoding="utf-8").strip().split()
        gt_rows.append(xywhn_to_xywh_pixels(label_parts, width, height))

    return [str(p.resolve()) for p in frames], np.asarray(gt_rows, dtype=np.float64)


def setup_pytracking(project_root: Path) -> None:
    pytracking_dir = project_root / "MyECOTracker" / "pytracking"
    if str(pytracking_dir) not in sys.path:
        sys.path.insert(0, str(pytracking_dir))


def calc_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def calc_center_error(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
    cv2.putText(
        image,
        label,
        (max(0, x), max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def export_overlays(
    frame_paths: list[str],
    gt: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
    save_frames: int,
    save_video: bool,
) -> None:
    preview_dir = out_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(save_frames, len(frame_paths))
    sample_indices = np.linspace(0, len(frame_paths) - 1, num=sample_count, dtype=int)
    sample_set = set(int(i) for i in sample_indices.tolist())

    writer = None
    if save_video:
        first = cv2.imread(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Failed to read {frame_paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_dir / "preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (width, height),
        )

    for idx, frame_path in enumerate(frame_paths):
        image = cv2.imread(frame_path)
        if image is None:
            raise RuntimeError(f"Failed to read {frame_path}")

        draw_box(image, gt[idx], (0, 255, 0), "GT")
        draw_box(image, pred[idx], (0, 0, 255), "ECO")

        iou = calc_iou(gt[idx], pred[idx])
        ce = calc_center_error(gt[idx], pred[idx])
        cv2.putText(
            image,
            f"frame={idx} IoU={iou:.3f} CE={ce:.1f}px",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if idx in sample_set:
            cv2.imwrite(str(preview_dir / f"frame_{idx:06d}.png"), image)

        if writer is not None:
            writer.write(image)

    if writer is not None:
        writer.release()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    dataset_root = (project_root / args.dataset_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, gt = load_sequence(dataset_root, args.max_frames)
    setup_pytracking(project_root)

    from pytracking.evaluation.data import Sequence
    from pytracking.evaluation.tracker import Tracker

    sequence_name = dataset_root.name
    sequence = Sequence(sequence_name, frame_paths, "datatest", gt.copy())
    tracker = Tracker(args.tracker_name, args.param)
    output = tracker.run_sequence(sequence, debug=0, visdom_info={"use_visdom": False})

    pred = np.asarray(output["target_bbox"], dtype=np.float64)
    times = np.asarray(output["time"], dtype=np.float64)
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape}")

    ious = np.asarray([calc_iou(gt_i, pred_i) for gt_i, pred_i in zip(gt, pred)], dtype=np.float64)
    center_errors = np.asarray([calc_center_error(gt_i, pred_i) for gt_i, pred_i in zip(gt, pred)], dtype=np.float64)

    success_auc = float(np.mean(ious))
    precision_20px = float(np.mean(center_errors <= 20.0))
    fps = float(len(times) / np.sum(times))

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"frames={len(frame_paths)}\n")
        f.write(f"tracker={args.tracker_name}\n")
        f.write(f"param={args.param}\n")
        f.write(f"success_auc_iou_mean={success_auc:.6f}\n")
        f.write(f"precision_20px={precision_20px:.6f}\n")
        f.write(f"avg_center_error_px={float(np.mean(center_errors)):.6f}\n")
        f.write(f"avg_iou={float(np.mean(ious)):.6f}\n")
        f.write(f"fps={fps:.6f}\n")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "frame_path",
                "gt_x",
                "gt_y",
                "gt_w",
                "gt_h",
                "pred_x",
                "pred_y",
                "pred_w",
                "pred_h",
                "iou",
                "center_error_px",
                "time_s",
            ]
        )
        for idx, frame_path in enumerate(frame_paths):
            writer.writerow(
                [
                    idx,
                    frame_path,
                    *gt[idx].tolist(),
                    *pred[idx].tolist(),
                    float(ious[idx]),
                    float(center_errors[idx]),
                    float(times[idx]),
                ]
            )

    export_overlays(
        frame_paths=frame_paths,
        gt=gt,
        pred=pred,
        out_dir=output_dir,
        save_frames=args.save_frames,
        save_video=args.save_video,
    )

    print(f"dataset_root={dataset_root}")
    print(f"frames={len(frame_paths)}")
    print(f"tracker={args.tracker_name}")
    print(f"param={args.param}")
    print(f"success_auc_iou_mean={success_auc:.6f}")
    print(f"precision_20px={precision_20px:.6f}")
    print(f"avg_center_error_px={float(np.mean(center_errors)):.6f}")
    print(f"fps={fps:.6f}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
