#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SUCCESS_THRESHOLDS = np.arange(0.0, 1.0001, 0.05, dtype=np.float64)
PRECISION_THRESHOLDS = np.arange(0, 51, 1, dtype=np.float64)


@dataclass
class SequenceData:
    video_path: Path
    frame_count: int
    frame_width: int
    frame_height: int
    video_fps: float
    gt_xywh: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MyECOTracker and CSRT on a labeled video sequence."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("datatest") / "test1" / "img" / "1.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("datatest") / "test1" / "labels",
        help="Directory containing YOLO-format frame_XXXXXX.txt labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "eval_datatest_test1",
        help="Directory for summary CSVs and per-tracker predictions.",
    )
    parser.add_argument("--tracker-name", default="eco", help="Tracker module name.")
    parser.add_argument(
        "--param",
        default="verified_otb936",
        help="Tracker parameter alias under pytracking.parameter.<tracker-name>.",
    )
    return parser.parse_args()


def setup_pytracking(project_root: Path) -> None:
    pytracking_dir = project_root / "MyECOTracker" / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def clip_xywh(box_xywh: Sequence[float], width: int, height: int) -> list[float]:
    x, y, bw, bh = [float(v) for v in box_xywh]
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    bw = max(1.0, min(bw, width - x))
    bh = max(1.0, min(bh, height - y))
    return [x, y, bw, bh]


def yolo_label_to_xywh(label_line: str, width: int, height: int) -> list[float]:
    parts = label_line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO columns, got {len(parts)}: {label_line}")
    _class_id, xc, yc, bw, bh = map(float, parts)
    x = (xc - bw / 2.0) * width
    y = (yc - bh / 2.0) * height
    return clip_xywh([x, y, bw * width, bh * height], width=width, height=height)


def load_sequence(video_path: Path, labels_dir: Path) -> SequenceData:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        video_fps = 30.0
    cap.release()

    label_files = sorted(labels_dir.glob("frame_*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No frame_*.txt labels found in {labels_dir}")

    gt_rows: list[list[float]] = []
    for label_path in label_files:
        label_line = label_path.read_text(encoding="utf-8").strip()
        gt_rows.append(yolo_label_to_xywh(label_line, width=frame_width, height=frame_height))

    return SequenceData(
        video_path=video_path,
        frame_count=len(gt_rows),
        frame_width=frame_width,
        frame_height=frame_height,
        video_fps=video_fps,
        gt_xywh=np.asarray(gt_rows, dtype=np.float64),
    )


def calc_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
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


def calc_center_error(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def compute_metrics(gt: np.ndarray, pred: np.ndarray, times_s: np.ndarray) -> dict[str, float]:
    ious = np.asarray([calc_iou(gt_i, pred_i) for gt_i, pred_i in zip(gt, pred)], dtype=np.float64)
    center_errors = np.asarray([calc_center_error(gt_i, pred_i) for gt_i, pred_i in zip(gt, pred)], dtype=np.float64)
    success_curve = np.asarray([np.mean(ious >= thr) for thr in SUCCESS_THRESHOLDS], dtype=np.float64)
    precision_curve = np.asarray([np.mean(center_errors <= thr) for thr in PRECISION_THRESHOLDS], dtype=np.float64)

    total_time = float(np.sum(times_s))
    fps = float(len(times_s) / total_time) if total_time > 0 else float("nan")
    return {
        "frames": float(len(gt)),
        "auc": float(np.mean(success_curve) * 100.0),
        "success50": float(np.mean(ious >= 0.50) * 100.0),
        "success75": float(np.mean(ious >= 0.75) * 100.0),
        "precision20": float(np.mean(center_errors <= 20.0) * 100.0),
        "mean_iou": float(np.mean(ious)),
        "mean_center_error": float(np.mean(center_errors)),
        "fps": fps,
    }


def write_prediction_csv(path: Path, gt: np.ndarray, pred: np.ndarray, times_s: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
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
        for idx, (gt_box, pred_box, time_s) in enumerate(zip(gt, pred, times_s)):
            writer.writerow(
                [
                    idx,
                    *gt_box.tolist(),
                    *pred_box.tolist(),
                    calc_iou(gt_box, pred_box),
                    calc_center_error(gt_box, pred_box),
                    float(time_s),
                ]
            )


def create_myeco_tracker(project_root: Path, tracker_name: str, param_name: str):
    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    wrapper = Tracker(tracker_name, param_name)
    params = wrapper.get_parameters()
    params.debug = 0
    params.visualization = False
    tracker = wrapper.create_tracker(params)
    if hasattr(tracker, "initialize_features"):
        tracker.initialize_features()
    return tracker


def run_myeco(project_root: Path, seq: SequenceData, tracker_name: str, param_name: str) -> tuple[np.ndarray, np.ndarray]:
    tracker = create_myeco_tracker(project_root, tracker_name, param_name)
    cap = cv2.VideoCapture(str(seq.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for MyECO: {seq.video_path}")

    ok, first_frame_bgr = cap.read()
    if not ok or first_frame_bgr is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {seq.video_path}")

    gt_init = seq.gt_xywh[0].tolist()
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    out = tracker.initialize(first_frame_rgb, {"init_bbox": list(map(float, gt_init))}) or {}
    init_elapsed = time.perf_counter() - start
    prev_output = dict(out)

    pred_rows: list[list[float]] = [[float(v) for v in out.get("target_bbox", gt_init)]]
    time_rows: list[float] = [init_elapsed]

    while len(pred_rows) < seq.frame_count:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
        elapsed = time.perf_counter() - start
        prev_output = dict(out)
        pred_rows.append([float(v) for v in out["target_bbox"]])
        time_rows.append(elapsed)

    cap.release()
    pred = np.asarray(pred_rows, dtype=np.float64)
    times_s = np.asarray(time_rows, dtype=np.float64)
    if pred.shape[0] != seq.frame_count:
        raise RuntimeError(f"MyECO processed {pred.shape[0]} frames, expected {seq.frame_count}")
    return pred, times_s


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV CSRT tracker is not available in this environment.")


def run_csrt(seq: SequenceData) -> tuple[np.ndarray, np.ndarray]:
    tracker = create_csrt_tracker()
    cap = cv2.VideoCapture(str(seq.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for CSRT: {seq.video_path}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {seq.video_path}")

    gt_init = seq.gt_xywh[0]
    init_box = tuple(int(round(v)) for v in gt_init.tolist())
    tracker.init(first_frame, init_box)

    pred_rows: list[list[float]] = [clip_xywh(gt_init.tolist(), width=seq.frame_width, height=seq.frame_height)]
    time_rows: list[float] = [0.0]

    while len(pred_rows) < seq.frame_count:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        start = time.perf_counter()
        success, box = tracker.update(frame)
        elapsed = time.perf_counter() - start
        if success:
            pred_rows.append(clip_xywh(box, width=seq.frame_width, height=seq.frame_height))
        else:
            pred_rows.append(pred_rows[-1].copy())
        time_rows.append(elapsed)

    cap.release()
    pred = np.asarray(pred_rows, dtype=np.float64)
    times_s = np.asarray(time_rows, dtype=np.float64)
    if pred.shape[0] != seq.frame_count:
        raise RuntimeError(f"CSRT processed {pred.shape[0]} frames, expected {seq.frame_count}")
    return pred, times_s


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tracker",
                "frames",
                "auc",
                "success50",
                "success75",
                "precision20",
                "mean_iou",
                "mean_center_error",
                "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    video_path = (project_root / args.video).resolve() if not args.video.is_absolute() else args.video.resolve()
    labels_dir = (project_root / args.labels_dir).resolve() if not args.labels_dir.is_absolute() else args.labels_dir.resolve()
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seq = load_sequence(video_path=video_path, labels_dir=labels_dir)

    myeco_pred, myeco_times = run_myeco(project_root, seq, tracker_name=args.tracker_name, param_name=args.param)
    csrt_pred, csrt_times = run_csrt(seq)

    myeco_metrics = compute_metrics(seq.gt_xywh, myeco_pred, myeco_times)
    csrt_metrics = compute_metrics(seq.gt_xywh, csrt_pred, csrt_times)

    write_prediction_csv(output_dir / "myeco_predictions.csv", seq.gt_xywh, myeco_pred, myeco_times)
    write_prediction_csv(output_dir / "csrt_predictions.csv", seq.gt_xywh, csrt_pred, csrt_times)

    summary_rows: list[dict[str, float | str]] = [
        {"tracker": f"{args.tracker_name}_{args.param}", **myeco_metrics},
        {"tracker": "opencv_csrt", **csrt_metrics},
    ]
    write_summary(output_dir / "summary.csv", summary_rows)

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"video_path={video_path}\n")
        f.write(f"labels_dir={labels_dir}\n")
        f.write(f"frames_evaluated={seq.frame_count}\n")
        f.write(f"frame_width={seq.frame_width}\n")
        f.write(f"frame_height={seq.frame_height}\n")
        f.write(f"video_fps={seq.video_fps:.6f}\n")
        for row in summary_rows:
            f.write(
                (
                    f"{row['tracker']}: "
                    f"AUC={float(row['auc']):.4f} "
                    f"Success50={float(row['success50']):.4f} "
                    f"Success75={float(row['success75']):.4f} "
                    f"Precision20={float(row['precision20']):.4f} "
                    f"MeanIoU={float(row['mean_iou']):.6f} "
                    f"MeanCE={float(row['mean_center_error']):.4f} "
                    f"FPS={float(row['fps']):.4f}\n"
                )
            )

    for row in summary_rows:
        print(
            " | ".join(
                [
                    f"tracker={row['tracker']}",
                    f"AUC={float(row['auc']):.4f}",
                    f"Success50={float(row['success50']):.4f}",
                    f"Success75={float(row['success75']):.4f}",
                    f"Precision20={float(row['precision20']):.4f}",
                    f"MeanIoU={float(row['mean_iou']):.6f}",
                    f"FPS={float(row['fps']):.4f}",
                ]
            )
        )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
