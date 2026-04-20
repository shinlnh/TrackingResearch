#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MyECOTracker on a local video and export non-interactive diagnostics."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video.")
    parser.add_argument("--tracker-name", default="eco", help="Tracker module name.")
    parser.add_argument(
        "--param",
        default="verified_otb936",
        help="Tracker parameter alias under pytracking.parameter.<tracker-name>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to runs/myecotracker_video_<video_stem>.",
    )
    parser.add_argument(
        "--init-xywh",
        nargs=4,
        type=float,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Manual init box in XYWH pixels. If omitted, a person box is auto-detected on the first frame.",
    )
    parser.add_argument(
        "--detector-model",
        default="yolov8n.pt",
        help="Ultralytics model used for automatic first-frame person detection.",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=1280,
        help="Image size for automatic person detection.",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for automatic person detection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--save-preview-frames",
        type=int,
        default=12,
        help="Number of evenly sampled annotated frames to save.",
    )
    parser.add_argument(
        "--save-all-frames",
        action="store_true",
        help="Also export every annotated frame as a JPG.",
    )
    return parser.parse_args()


def setup_pytracking(project_root: Path) -> None:
    pytracking_dir = project_root / "MyECOTracker" / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def sanitize_stem(stem: str) -> str:
    safe = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem]
    return "".join(safe).strip("_") or "video"


def clip_xywh(box_xywh: Sequence[float], frame_shape: tuple[int, int, int]) -> list[float]:
    h, w = frame_shape[:2]
    x, y, bw, bh = [float(v) for v in box_xywh]
    x = max(0.0, min(x, w - 1.0))
    y = max(0.0, min(y, h - 1.0))
    bw = max(1.0, min(bw, w - x))
    bh = max(1.0, min(bh, h - y))
    return [x, y, bw, bh]


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def draw_box(image: np.ndarray, box: Sequence[float], color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(float(v))) for v in box]
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


def draw_header(image: np.ndarray, frame_index: int, score: float, elapsed_s: float) -> None:
    score_text = "nan" if math.isnan(score) else f"{score:.3f}"
    lines = [
        f"frame={frame_index}",
        f"score={score_text}",
        f"time={elapsed_s * 1000.0:.1f}ms",
    ]
    x, y = 18, 18
    width = 260
    line_height = 24
    height = 12 + line_height * len(lines)
    cv2.rectangle(image, (x, y), (x + width, y + height), (24, 24, 24), -1)
    for idx, line in enumerate(lines):
        baseline_y = y + 28 + idx * line_height
        cv2.putText(
            image,
            line,
            (x + 10, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 320) -> None:
    if not image_paths:
        return

    thumbs: list[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = thumb_width / float(w)
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(h * scale)))))
        label_h = 28
        canvas = np.full((thumb.shape[0] + label_h, thumb.shape[1], 3), 16, dtype=np.uint8)
        canvas[: thumb.shape[0], : thumb.shape[1]] = thumb
        cv2.putText(
            canvas,
            image_path.stem,
            (10, thumb.shape[0] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(canvas)

    if not thumbs:
        return

    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    cell_h = max(img.shape[0] for img in thumbs)
    cell_w = max(img.shape[1] for img in thumbs)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 8, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * cell_h
        x0 = col * cell_w
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb

    cv2.imwrite(str(out_path), sheet)


def detect_person_bbox(frame_bgr: np.ndarray, model_name: str, imgsz: int, conf: float) -> tuple[list[float], float]:
    detector = YOLO(model_name)
    result = detector.predict(
        source=frame_bgr,
        verbose=False,
        classes=[0],
        conf=conf,
        imgsz=imgsz,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("No person detected on the first frame.")

    boxes_xyxy = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
    return xyxy_to_xywh(boxes_xyxy[best_idx]), float(scores[best_idx])


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("runs") / f"myecotracker_video_{sanitize_stem(video_path.stem)}"
    output_dir = (project_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_frames_dir = output_dir / "all_frames_overlay"
    if args.save_all_frames:
        all_frames_dir.mkdir(parents=True, exist_ok=True)

    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, first_frame_bgr = cap.read()
    if not ok or first_frame_bgr is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    frame_height, frame_width = first_frame_bgr.shape[:2]
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.init_xywh is not None:
        init_bbox = clip_xywh(args.init_xywh, first_frame_bgr.shape)
        init_score = float("nan")
        init_source = "manual"
    else:
        init_bbox_detected, init_score = detect_person_bbox(
            first_frame_bgr,
            model_name=args.detector_model,
            imgsz=args.detector_imgsz,
            conf=args.detector_conf,
        )
        init_bbox = clip_xywh(init_bbox_detected, first_frame_bgr.shape)
        init_source = f"yolo:{args.detector_model}"

    first_frame_annotated = first_frame_bgr.copy()
    draw_box(first_frame_annotated, init_bbox, (0, 255, 255), "INIT")
    draw_header(first_frame_annotated, frame_index=0, score=init_score, elapsed_s=0.0)
    cv2.imwrite(str(output_dir / "init_frame.png"), first_frame_annotated)

    tracker_wrapper = Tracker(args.tracker_name, args.param)
    params = tracker_wrapper.get_parameters()
    params.debug = 0
    params.visualization = False
    tracker = tracker_wrapper.create_tracker(params)
    if hasattr(tracker, "initialize_features"):
        tracker.initialize_features()

    writer = cv2.VideoWriter(
        str(output_dir / "tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to create output video writer.")

    if args.max_frames is not None:
        target_frames = max(1, min(frame_count_reported if frame_count_reported > 0 else args.max_frames, args.max_frames))
    else:
        target_frames = frame_count_reported if frame_count_reported > 0 else None

    if target_frames is None or target_frames <= 0:
        sample_indices = {0}
    else:
        sample_count = min(max(1, args.save_preview_frames), target_frames)
        sample_indices = set(int(v) for v in np.linspace(0, target_frames - 1, num=sample_count, dtype=int).tolist())

    predictions_path = output_dir / "predictions.csv"
    preview_written: list[Path] = []
    frame_index = 0
    total_track_time = 0.0
    tracker_scores: list[float] = []

    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    init_out = tracker.initialize(first_frame_rgb, {"init_bbox": list(map(float, init_bbox))}) or {}
    init_elapsed = time.perf_counter() - start
    prev_output = dict(init_out)
    init_track_score = float(getattr(tracker, "last_max_score", float("nan")))
    tracker_scores.append(init_track_score)
    total_track_time += init_elapsed

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_index", "pred_x", "pred_y", "pred_w", "pred_h", "score", "time_s"])

        current_box = [float(v) for v in init_out.get("target_bbox", init_bbox)]
        csv_writer.writerow([frame_index, *current_box, init_track_score, init_elapsed])

        first_overlay = first_frame_bgr.copy()
        draw_box(first_overlay, current_box, (0, 0, 255), "ECO")
        draw_header(first_overlay, frame_index=frame_index, score=init_track_score, elapsed_s=init_elapsed)
        writer.write(first_overlay)

        if frame_index in sample_indices:
            preview_path = preview_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(preview_path), first_overlay)
            preview_written.append(preview_path)

        if args.save_all_frames:
            cv2.imwrite(str(all_frames_dir / f"frame_{frame_index:06d}.jpg"), first_overlay)

        while True:
            if target_frames is not None and frame_index + 1 >= target_frames:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            frame_index += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            start = time.perf_counter()
            out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
            elapsed = time.perf_counter() - start
            prev_output = dict(out)
            pred_box = [float(v) for v in out["target_bbox"]]
            score = float(getattr(tracker, "last_max_score", float("nan")))

            tracker_scores.append(score)
            total_track_time += elapsed
            csv_writer.writerow([frame_index, *pred_box, score, elapsed])

            overlay = frame_bgr.copy()
            draw_box(overlay, pred_box, (0, 0, 255), "ECO")
            draw_header(overlay, frame_index=frame_index, score=score, elapsed_s=elapsed)
            writer.write(overlay)

            if frame_index in sample_indices:
                preview_path = preview_dir / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(preview_path), overlay)
                preview_written.append(preview_path)

            if args.save_all_frames:
                cv2.imwrite(str(all_frames_dir / f"frame_{frame_index:06d}.jpg"), overlay)

    cap.release()
    writer.release()

    if preview_written:
        make_contact_sheet(preview_written, output_dir / "contact_sheet.png")

    boxes = np.loadtxt(predictions_path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    boxes = np.atleast_2d(boxes)
    scores_np = np.asarray(tracker_scores, dtype=np.float64)
    valid_scores = scores_np[np.isfinite(scores_np)]
    tracker_fps = float((frame_index + 1) / total_track_time) if total_track_time > 0 else float("nan")

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"video_path={video_path}\n")
        f.write(f"frames_processed={frame_index + 1}\n")
        f.write(f"reported_frame_count={frame_count_reported}\n")
        f.write(f"video_fps={video_fps:.6f}\n")
        f.write(f"frame_width={frame_width}\n")
        f.write(f"frame_height={frame_height}\n")
        f.write(f"tracker={args.tracker_name}\n")
        f.write(f"param={args.param}\n")
        f.write(f"init_source={init_source}\n")
        f.write(f"init_bbox_xywh={','.join(f'{v:.4f}' for v in init_bbox)}\n")
        if math.isnan(init_score):
            f.write("init_detection_conf=nan\n")
        else:
            f.write(f"init_detection_conf={init_score:.6f}\n")
        if valid_scores.size > 0:
            f.write(f"mean_tracker_score={float(np.mean(valid_scores)):.6f}\n")
            f.write(f"min_tracker_score={float(np.min(valid_scores)):.6f}\n")
            f.write(f"max_tracker_score={float(np.max(valid_scores)):.6f}\n")
        else:
            f.write("mean_tracker_score=nan\n")
            f.write("min_tracker_score=nan\n")
            f.write("max_tracker_score=nan\n")
        f.write(f"tracker_fps={tracker_fps:.6f}\n")
        f.write(f"mean_pred_w={float(np.mean(boxes[:, 2])):.6f}\n")
        f.write(f"mean_pred_h={float(np.mean(boxes[:, 3])):.6f}\n")
        f.write(f"mean_pred_area={float(np.mean(boxes[:, 2] * boxes[:, 3])):.6f}\n")

    print(f"video_path={video_path}")
    print(f"frames_processed={frame_index + 1}")
    print(f"video_fps={video_fps:.6f}")
    print(f"tracker_fps={tracker_fps:.6f}")
    print(f"init_source={init_source}")
    print(f"init_bbox_xywh={','.join(f'{v:.2f}' for v in init_bbox)}")
    if valid_scores.size > 0:
        print(f"mean_tracker_score={float(np.mean(valid_scores)):.6f}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
