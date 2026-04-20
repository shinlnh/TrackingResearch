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
from ultralytics import YOLO


@dataclass
class Detection:
    box_xywh: list[float]
    conf: float


def default_yolo_model(project_root: Path) -> str:
    preferred = project_root / "MyPersonTracker" / "weights" / "yolo_person_only.pt"
    if preferred.is_file():
        return str(preferred)
    return "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple YOLO + MyECO hybrid on a local video."
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
        help="Output directory. Defaults to runs/myecotracker_yolo_hybrid_<video_stem>.",
    )
    parser.add_argument(
        "--detector-model",
        default=None,
        help="Ultralytics model used for person detection. Defaults to person-only weights if present.",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=1280,
        help="Image size for YOLO inference.",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detection.",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=5,
        help="Run YOLO every N frames.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.72,
        help="Force YOLO correction when MyECO score drops below this value.",
    )
    parser.add_argument(
        "--reinit-iou-threshold",
        type=float,
        default=0.55,
        help="Hard reinit when tracker and detector IoU fall below this value.",
    )
    parser.add_argument(
        "--size-ratio-threshold",
        type=float,
        default=1.45,
        help="Hard reinit when tracker and detector area ratio exceeds this factor.",
    )
    parser.add_argument(
        "--detector-weight",
        type=float,
        default=0.65,
        help="Weight of YOLO box when applying a soft correction.",
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


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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


def center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def area_ratio(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    _, _, aw, ah = [float(v) for v in box_a]
    _, _, bw, bh = [float(v) for v in box_b]
    area_a = max(1.0, aw * ah)
    area_b = max(1.0, bw * bh)
    return max(area_a, area_b) / min(area_a, area_b)


def blend_boxes(box_a: Sequence[float], box_b: Sequence[float], box_b_weight: float) -> list[float]:
    alpha = float(box_b_weight)
    beta = 1.0 - alpha
    return [beta * float(a) + alpha * float(b) for a, b in zip(box_a, box_b)]


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


def draw_header(image: np.ndarray, frame_index: int, score: float, mode: str, det_conf: float | None) -> None:
    score_text = "nan" if math.isnan(score) else f"{score:.3f}"
    conf_text = "-" if det_conf is None or math.isnan(det_conf) else f"{det_conf:.3f}"
    lines = [
        f"frame={frame_index}",
        f"eco_score={score_text}",
        f"mode={mode}",
        f"det_conf={conf_text}",
    ]
    x, y = 18, 18
    width = 280
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


def detect_people(frame_bgr: np.ndarray, detector: YOLO, imgsz: int, conf: float) -> list[Detection]:
    result = detector.predict(
        source=frame_bgr,
        verbose=False,
        classes=[0],
        conf=conf,
        imgsz=imgsz,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    detections: list[Detection] = []
    for box_xyxy, score in zip(boxes_xyxy, scores):
        detections.append(Detection(box_xywh=xyxy_to_xywh(box_xyxy), conf=float(score)))
    return detections


def select_detection(
    detections: Sequence[Detection],
    reference_box: Sequence[float] | None,
    frame_shape: tuple[int, int, int],
) -> Detection | None:
    if not detections:
        return None
    if reference_box is None:
        return max(detections, key=lambda det: det.conf)

    h, w = frame_shape[:2]
    diag = float(np.hypot(w, h))

    def rank(det: Detection) -> tuple[float, float]:
        iou = box_iou(reference_box, det.box_xywh)
        dist = center_distance(reference_box, det.box_xywh)
        proximity = max(0.0, 1.0 - dist / max(1.0, diag))
        score = 2.0 * iou + 0.6 * proximity + 0.3 * det.conf
        return score, det.conf

    return max(detections, key=rank)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("runs") / f"myecotracker_yolo_hybrid_{sanitize_stem(video_path.stem)}"
    output_dir = (project_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_frames_dir = output_dir / "all_frames_overlay"
    if args.save_all_frames:
        all_frames_dir.mkdir(parents=True, exist_ok=True)

    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    detector_model = args.detector_model or default_yolo_model(project_root)
    detector = YOLO(detector_model)

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

    init_detections = detect_people(first_frame_bgr, detector, imgsz=args.detector_imgsz, conf=args.detector_conf)
    init_detection = select_detection(init_detections, reference_box=None, frame_shape=first_frame_bgr.shape)
    if init_detection is None:
        cap.release()
        raise RuntimeError("No person detected on the first frame.")
    init_bbox = clip_xywh(init_detection.box_xywh, first_frame_bgr.shape)

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

    sample_count = min(max(1, args.save_preview_frames), max(1, frame_count_reported))
    sample_indices = set(int(v) for v in np.linspace(0, max(0, frame_count_reported - 1), num=sample_count, dtype=int).tolist())

    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    init_out = tracker.initialize(first_frame_rgb, {"init_bbox": list(map(float, init_bbox))}) or {}
    init_elapsed = time.perf_counter() - start

    prev_output = dict(init_out)
    tracker_box = [float(v) for v in init_out.get("target_bbox", init_bbox)]
    final_box = tracker_box.copy()
    tracker_score = float(getattr(tracker, "last_max_score", float("nan")))
    last_det_box = init_bbox.copy()

    predictions_path = output_dir / "predictions.csv"
    preview_written: list[Path] = []
    frame_index = 0
    total_track_time = init_elapsed
    total_detect_time = 0.0
    detection_calls = 1
    soft_corrections = 0
    hard_reinits = 0
    frames_with_detection = 1

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame_index",
                "tracker_x",
                "tracker_y",
                "tracker_w",
                "tracker_h",
                "final_x",
                "final_y",
                "final_w",
                "final_h",
                "det_x",
                "det_y",
                "det_w",
                "det_h",
                "tracker_score",
                "det_conf",
                "mode",
                "track_time_s",
                "detect_time_s",
            ]
        )

        overlay = first_frame_bgr.copy()
        draw_box(overlay, tracker_box, (0, 0, 255), "ECO raw")
        draw_box(overlay, init_bbox, (0, 180, 255), "YOLO")
        draw_box(overlay, final_box, (0, 255, 0), "Hybrid")
        draw_header(overlay, frame_index=0, score=tracker_score, mode="init", det_conf=init_detection.conf)
        writer.write(overlay)
        cv2.imwrite(str(output_dir / "init_frame.png"), overlay)
        if frame_index in sample_indices:
            preview_path = preview_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(preview_path), overlay)
            preview_written.append(preview_path)
        if args.save_all_frames:
            cv2.imwrite(str(all_frames_dir / f"frame_{frame_index:06d}.jpg"), overlay)

        csv_writer.writerow(
            [
                frame_index,
                *tracker_box,
                *final_box,
                *init_bbox,
                tracker_score,
                init_detection.conf,
                "init",
                init_elapsed,
                0.0,
            ]
        )

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            frame_index += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            track_start = time.perf_counter()
            out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
            track_elapsed = time.perf_counter() - track_start
            total_track_time += track_elapsed

            prev_output = dict(out)
            tracker_box = [float(v) for v in out["target_bbox"]]
            tracker_score = float(getattr(tracker, "last_max_score", float("nan")))
            final_box = tracker_box.copy()
            matched_det: Detection | None = None
            det_elapsed = 0.0
            mode = "track"

            need_detection = (frame_index % max(1, args.detect_interval) == 0) or (
                not math.isnan(tracker_score) and tracker_score < args.low_score_threshold
            )
            if need_detection:
                detect_start = time.perf_counter()
                detections = detect_people(frame_bgr, detector, imgsz=args.detector_imgsz, conf=args.detector_conf)
                det_elapsed = time.perf_counter() - detect_start
                total_detect_time += det_elapsed
                detection_calls += 1

                reference_box = last_det_box if last_det_box is not None else tracker_box
                matched_det = select_detection(detections, reference_box=reference_box, frame_shape=frame_bgr.shape)
                if matched_det is not None:
                    frames_with_detection += 1
                    det_box = clip_xywh(matched_det.box_xywh, frame_bgr.shape)
                    last_det_box = det_box.copy()
                    det_iou = box_iou(tracker_box, det_box)
                    det_area_ratio = area_ratio(tracker_box, det_box)
                    force_hard = (
                        det_iou < args.reinit_iou_threshold
                        or det_area_ratio > args.size_ratio_threshold
                        or (not math.isnan(tracker_score) and tracker_score < args.low_score_threshold)
                    )
                    if force_hard:
                        final_box = det_box.copy()
                        tracker.initialize(frame_rgb, {"init_bbox": list(map(float, final_box))})
                        prev_output = {"target_bbox": final_box.copy()}
                        hard_reinits += 1
                        mode = "hard_reinit"
                    else:
                        final_box = blend_boxes(tracker_box, det_box, args.detector_weight)
                        tracker.initialize(frame_rgb, {"init_bbox": list(map(float, final_box))})
                        prev_output = {"target_bbox": final_box.copy()}
                        soft_corrections += 1
                        mode = "soft_fuse"

            overlay = frame_bgr.copy()
            draw_box(overlay, tracker_box, (0, 0, 255), "ECO raw")
            if matched_det is not None:
                draw_box(overlay, matched_det.box_xywh, (0, 180, 255), "YOLO")
            draw_box(overlay, final_box, (0, 255, 0), "Hybrid")
            draw_header(
                overlay,
                frame_index=frame_index,
                score=tracker_score,
                mode=mode,
                det_conf=None if matched_det is None else matched_det.conf,
            )
            writer.write(overlay)

            if frame_index in sample_indices:
                preview_path = preview_dir / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(preview_path), overlay)
                preview_written.append(preview_path)

            if args.save_all_frames:
                cv2.imwrite(str(all_frames_dir / f"frame_{frame_index:06d}.jpg"), overlay)

            csv_writer.writerow(
                [
                    frame_index,
                    *tracker_box,
                    *final_box,
                    *(matched_det.box_xywh if matched_det is not None else [math.nan] * 4),
                    tracker_score,
                    math.nan if matched_det is None else matched_det.conf,
                    mode,
                    track_elapsed,
                    det_elapsed,
                ]
            )

    cap.release()
    writer.release()

    if preview_written:
        make_contact_sheet(preview_written, output_dir / "contact_sheet.png")

    rows = list(csv.DictReader(predictions_path.open("r", encoding="utf-8")))
    final_ws = np.asarray([float(r["final_w"]) for r in rows], dtype=np.float64)
    final_hs = np.asarray([float(r["final_h"]) for r in rows], dtype=np.float64)
    scores = np.asarray([float(r["tracker_score"]) for r in rows], dtype=np.float64)
    valid_scores = scores[np.isfinite(scores)]
    total_time = total_track_time + total_detect_time
    overall_fps = float(len(rows) / total_time) if total_time > 0 else float("nan")

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"video_path={video_path}\n")
        f.write(f"frames_processed={len(rows)}\n")
        f.write(f"reported_frame_count={frame_count_reported}\n")
        f.write(f"video_fps={video_fps:.6f}\n")
        f.write(f"frame_width={frame_width}\n")
        f.write(f"frame_height={frame_height}\n")
        f.write(f"tracker={args.tracker_name}\n")
        f.write(f"param={args.param}\n")
        f.write(f"detector_model={detector_model}\n")
        f.write(f"detect_interval={args.detect_interval}\n")
        f.write(f"low_score_threshold={args.low_score_threshold:.6f}\n")
        f.write(f"reinit_iou_threshold={args.reinit_iou_threshold:.6f}\n")
        f.write(f"size_ratio_threshold={args.size_ratio_threshold:.6f}\n")
        f.write(f"detector_weight={args.detector_weight:.6f}\n")
        f.write(f"init_bbox_xywh={','.join(f'{v:.4f}' for v in init_bbox)}\n")
        f.write(f"init_detection_conf={init_detection.conf:.6f}\n")
        f.write(f"detection_calls={detection_calls}\n")
        f.write(f"frames_with_detection={frames_with_detection}\n")
        f.write(f"soft_corrections={soft_corrections}\n")
        f.write(f"hard_reinits={hard_reinits}\n")
        if valid_scores.size > 0:
            f.write(f"mean_tracker_score={float(np.mean(valid_scores)):.6f}\n")
            f.write(f"min_tracker_score={float(np.min(valid_scores)):.6f}\n")
            f.write(f"max_tracker_score={float(np.max(valid_scores)):.6f}\n")
        else:
            f.write("mean_tracker_score=nan\n")
            f.write("min_tracker_score=nan\n")
            f.write("max_tracker_score=nan\n")
        f.write(f"track_time_total_s={total_track_time:.6f}\n")
        f.write(f"detect_time_total_s={total_detect_time:.6f}\n")
        f.write(f"overall_fps={overall_fps:.6f}\n")
        f.write(f"mean_final_w={float(np.mean(final_ws)):.6f}\n")
        f.write(f"mean_final_h={float(np.mean(final_hs)):.6f}\n")
        f.write(f"mean_final_area={float(np.mean(final_ws * final_hs)):.6f}\n")
        f.write(f"min_final_w={float(np.min(final_ws)):.6f}\n")
        f.write(f"min_final_h={float(np.min(final_hs)):.6f}\n")
        f.write(f"max_final_w={float(np.max(final_ws)):.6f}\n")
        f.write(f"max_final_h={float(np.max(final_hs)):.6f}\n")

    print(f"video_path={video_path}")
    print(f"frames_processed={len(rows)}")
    print(f"detection_calls={detection_calls}")
    print(f"soft_corrections={soft_corrections}")
    print(f"hard_reinits={hard_reinits}")
    print(f"overall_fps={overall_fps:.6f}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
