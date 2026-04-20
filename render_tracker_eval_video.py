#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a clean evaluation video from GT + prediction CSV."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to the source video.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction CSV from evaluate_video_trackers_gt.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional YOLO label directory for drawing GT when the prediction CSV does not contain GT columns.",
    )
    parser.add_argument(
        "--box-columns",
        nargs=4,
        default=("pred_x", "pred_y", "pred_w", "pred_h"),
        metavar=("X", "Y", "W", "H"),
        help="CSV columns for the box to render.",
    )
    parser.add_argument(
        "--gt-columns",
        nargs=4,
        default=("gt_x", "gt_y", "gt_w", "gt_h"),
        metavar=("GT_X", "GT_Y", "GT_W", "GT_H"),
        help="CSV columns for ground truth, when present.",
    )
    parser.add_argument(
        "--time-columns",
        nargs="+",
        default=("time_s",),
        metavar="TIME_COL",
        help="One or more CSV time columns used to estimate running FPS.",
    )
    parser.add_argument(
        "--fps-mode",
        choices=("cumulative", "instant"),
        default="cumulative",
        help="How to compute the FPS text from the provided time columns.",
    )
    parser.add_argument(
        "--hide-gt",
        action="store_true",
        help="Do not draw the ground-truth box.",
    )
    parser.add_argument(
        "--hide-pred",
        action="store_true",
        help="Do not draw the predicted box.",
    )
    return parser.parse_args()


def draw_box(image: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)


def draw_header(image: np.ndarray, frame_index: int, fps_value: float) -> None:
    text = f"frame={frame_index}  fps={fps_value:.2f}"
    cv2.putText(
        image,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def load_rows(predictions_path: Path) -> list[dict[str, str]]:
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_box(row: dict[str, str], columns: Sequence[str]) -> tuple[float, float, float, float]:
    return tuple(float(row[col]) for col in columns)


def has_columns(row: dict[str, str], columns: Sequence[str]) -> bool:
    return all(col in row for col in columns)


def yolo_label_to_xywh(label_line: str, width: int, height: int) -> tuple[float, float, float, float]:
    parts = label_line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO columns, got {len(parts)}: {label_line}")
    _class_id, xc, yc, bw, bh = map(float, parts)
    x = (xc - bw / 2.0) * width
    y = (yc - bh / 2.0) * height
    return (x, y, bw * width, bh * height)


def load_gt_boxes_from_labels(
    labels_dir: Path,
    rows: Sequence[dict[str, str]],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float] | None]:
    gt_boxes: list[tuple[float, float, float, float] | None] = []
    for row in rows:
        frame_index = int(row["frame_index"])
        label_path = labels_dir / f"frame_{frame_index:06d}.txt"
        if not label_path.exists():
            gt_boxes.append(None)
            continue
        label_line = label_path.read_text(encoding="utf-8").strip()
        gt_boxes.append(yolo_label_to_xywh(label_line, width=width, height=height))
    return gt_boxes


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve() if args.labels_dir is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(predictions_path)
    if not rows:
        raise RuntimeError(f"No prediction rows found in {predictions_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        video_fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gt_boxes_from_labels = None
    if labels_dir is not None:
        gt_boxes_from_labels = load_gt_boxes_from_labels(labels_dir, rows, width=frame_width, height=frame_height)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    elapsed_total = 0.0
    frame_count = 0

    try:
        for row_idx, row in enumerate(rows):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index = int(row["frame_index"])
            pred_box = read_box(row, args.box_columns)
            if gt_boxes_from_labels is not None:
                gt_box = gt_boxes_from_labels[row_idx]
            elif has_columns(row, args.gt_columns):
                gt_box = read_box(row, args.gt_columns)
            else:
                gt_box = None
            time_s = 0.0
            for time_col in args.time_columns:
                if time_col in row and row[time_col]:
                    time_s += max(0.0, float(row[time_col]))

            if not args.hide_gt and gt_box is not None:
                draw_box(frame, gt_box, (0, 255, 0))
            if not args.hide_pred:
                draw_box(frame, pred_box, (0, 0, 255))

            elapsed_total += time_s
            frame_count += 1
            if args.fps_mode == "instant":
                fps_value = (1.0 / time_s) if time_s > 0 else 0.0
            else:
                fps_value = (frame_count / elapsed_total) if elapsed_total > 0 else 0.0
            draw_header(frame, frame_index=frame_index, fps_value=fps_value)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
