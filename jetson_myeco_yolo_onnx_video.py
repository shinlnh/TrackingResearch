#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run MyECO + YOLO(ONNX/OpenCV DNN) on a video.")
    parser.add_argument("video_path", type=Path, help="Input video path.")
    parser.add_argument("--model-path", "--onnx-model", dest="model_path", type=Path, required=True, help="YOLO detector model path (.onnx or .torchscript).")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936", help="PyTracking parameter name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output video and csv.")
    parser.add_argument("--input-size", type=int, default=1280, help="YOLO ONNX input size.")
    parser.add_argument("--detector-backend", choices=["auto", "onnx", "torchscript"], default="auto", help="Detector backend.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device for torchscript backend.")
    parser.add_argument("--detector-conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.45, help="Detector NMS threshold.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Run detector every N frames.")
    parser.add_argument("--low-score-threshold", type=float, default=0.72, help="Force detector correction below this score.")
    parser.add_argument("--reinit-iou-threshold", type=float, default=0.55, help="Hard reinit below this IoU.")
    parser.add_argument("--size-ratio-threshold", type=float, default=1.45, help="Hard reinit above this area ratio.")
    parser.add_argument("--detector-weight", type=float, default=0.65, help="Soft fuse weight of detector box.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for smoke tests. 0 means full video.")
    return parser.parse_args()


def resolve_project_root(script_path):
    for candidate in [script_path] + list(script_path.parents):
        if (candidate / "pytracking").is_dir():
            return candidate
        if (candidate / "MyECOTracker" / "pytracking").is_dir():
            return candidate / "MyECOTracker"
    raise RuntimeError("Could not locate pytracking project root from %s" % script_path)


def setup_pytracking(project_root):
    pytracking_dir = project_root / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def clip_xywh(box_xywh, frame_shape):
    height, width = frame_shape[:2]
    x, y, bw, bh = [float(v) for v in box_xywh]
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    bw = max(1.0, min(bw, width - x))
    bh = max(1.0, min(bh, height - y))
    return [x, y, bw, bh]


def box_iou(box_a, box_b):
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


def center_distance(box_a, box_b):
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def area_ratio(box_a, box_b):
    _, _, aw, ah = [float(v) for v in box_a]
    _, _, bw, bh = [float(v) for v in box_b]
    area_a = max(1.0, aw * ah)
    area_b = max(1.0, bw * bh)
    return max(area_a, area_b) / min(area_a, area_b)


def blend_boxes(box_a, box_b, box_b_weight):
    alpha = float(box_b_weight)
    beta = 1.0 - alpha
    return [beta * float(a) + alpha * float(b) for a, b in zip(box_a, box_b)]


def draw_box(image, box, color):
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)


def draw_header(image, frame_index, fps_value):
    text = "frame=%d  fps=%.2f" % (frame_index, fps_value)
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


def resolve_backend(model_path, backend):
    if backend != "auto":
        return backend
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".torchscript":
        return "torchscript"
    raise RuntimeError("Could not infer detector backend from %s" % model_path)


def load_detector(model_path, backend, device):
    backend = resolve_backend(model_path, backend)
    if backend == "onnx":
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return {"backend": backend, "net": net}

    import torch

    if device == "auto":
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        torch_device = device
    model = torch.jit.load(str(model_path), map_location=torch_device)
    model.eval()
    return {"backend": backend, "model": model, "device": torch_device}


def detect_people(detector, frame_bgr, input_size, conf_thresh, nms_thresh):
    orig_h, orig_w = frame_bgr.shape[:2]
    if detector["backend"] == "onnx":
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255.0, (input_size, input_size), swapRB=True, crop=False)
        detector["net"].setInput(blob)
        out = detector["net"].forward()[0]
    else:
        import torch

        resized = cv2.resize(frame_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(detector["device"])
        with torch.no_grad():
            out = detector["model"](tensor)
        out = out.detach().cpu().numpy()[0]

    boxes = []
    scores = []
    for i in range(out.shape[1]):
        xc, yc, bw, bh, conf = [float(v) for v in out[:, i]]
        if conf < conf_thresh:
            continue
        x = (xc - bw / 2.0) * orig_w / float(input_size)
        y = (yc - bh / 2.0) * orig_h / float(input_size)
        box_w = bw * orig_w / float(input_size)
        box_h = bh * orig_h / float(input_size)
        boxes.append([x, y, box_w, box_h])
        scores.append(conf)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
    if indices is None or len(indices) == 0:
        return []

    indices = np.array(indices).reshape(-1).tolist()
    detections = []
    for idx in indices:
        detections.append({
            "box_xywh": clip_xywh(boxes[idx], frame_bgr.shape),
            "conf": float(scores[idx]),
        })
    return detections


def select_detection(detections, reference_box, frame_shape):
    if not detections:
        return None
    if reference_box is None:
        return max(detections, key=lambda det: det["conf"])

    height, width = frame_shape[:2]
    diag = float(np.hypot(width, height))

    def rank(det):
        iou = box_iou(reference_box, det["box_xywh"])
        dist = center_distance(reference_box, det["box_xywh"])
        proximity = max(0.0, 1.0 - dist / max(1.0, diag))
        return 2.0 * iou + 0.6 * proximity + 0.3 * det["conf"]

    return max(detections, key=rank)


def create_tracker(project_root, tracker_name, param_name):
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


def main():
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    video_path = args.video_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector(model_path, args.detector_backend, args.device)
    tracker = create_tracker(project_root, args.tracker_name, args.param)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: %s" % video_path)

    ok, first_frame_bgr = cap.read()
    if not ok or first_frame_bgr is None:
        cap.release()
        raise RuntimeError("Failed to read first frame from %s" % video_path)

    frame_height, frame_width = first_frame_bgr.shape[:2]
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        video_fps = 30.0
    frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detect_start = time.perf_counter()
    init_detections = detect_people(detector, first_frame_bgr, args.input_size, args.detector_conf, args.nms_threshold)
    init_detect_elapsed = time.perf_counter() - detect_start
    init_det = select_detection(init_detections, None, first_frame_bgr.shape)
    if init_det is None:
        cap.release()
        raise RuntimeError("No person detected on the first frame.")

    init_box = init_det["box_xywh"]
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    out = tracker.initialize(first_frame_rgb, {"init_bbox": list(map(float, init_box))}) or {}
    init_track_elapsed = time.perf_counter() - start
    prev_output = dict(out)

    tracker_box = [float(v) for v in out.get("target_bbox", init_box)]
    final_box = tracker_box[:]
    tracker_score = float(getattr(tracker, "last_max_score", float("nan")))
    last_det_box = init_box[:]

    writer = cv2.VideoWriter(
        str(output_dir / "tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to create output video.")

    predictions_path = output_dir / "predictions.csv"
    total_track_time = init_track_elapsed
    total_detect_time = init_detect_elapsed
    detection_calls = 1
    soft_corrections = 0
    hard_reinits = 0
    frame_index = 0

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow([
            "frame_index",
            "tracker_x", "tracker_y", "tracker_w", "tracker_h",
            "final_x", "final_y", "final_w", "final_h",
            "det_x", "det_y", "det_w", "det_h",
            "tracker_score", "det_conf", "mode",
            "track_time_s", "detect_time_s",
        ])

        overlay = first_frame_bgr.copy()
        total_elapsed = total_track_time + total_detect_time
        fps_value = (1.0 / total_elapsed) if total_elapsed > 0 else 0.0
        draw_box(overlay, final_box, (0, 255, 0))
        draw_header(overlay, frame_index, fps_value)
        writer.write(overlay)

        writer_csv.writerow([
            frame_index,
            tracker_box[0], tracker_box[1], tracker_box[2], tracker_box[3],
            final_box[0], final_box[1], final_box[2], final_box[3],
            init_box[0], init_box[1], init_box[2], init_box[3],
            tracker_score, init_det["conf"], "init",
            init_track_elapsed, init_detect_elapsed,
        ])

        while True:
            if args.max_frames > 0 and (frame_index + 1) >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_index += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            start = time.perf_counter()
            out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
            track_elapsed = time.perf_counter() - start
            total_track_time += track_elapsed
            prev_output = dict(out)

            tracker_box = [float(v) for v in out["target_bbox"]]
            final_box = tracker_box[:]
            tracker_score = float(getattr(tracker, "last_max_score", float("nan")))

            matched_det = None
            detect_elapsed = 0.0
            mode = "track"

            need_detection = (frame_index % max(1, args.detect_interval) == 0)
            if not math.isnan(tracker_score) and tracker_score < args.low_score_threshold:
                need_detection = True

            if need_detection:
                detect_start = time.perf_counter()
                detections = detect_people(detector, frame_bgr, args.input_size, args.detector_conf, args.nms_threshold)
                detect_elapsed = time.perf_counter() - detect_start
                total_detect_time += detect_elapsed
                detection_calls += 1

                reference_box = last_det_box if last_det_box is not None else tracker_box
                matched_det = select_detection(detections, reference_box, frame_bgr.shape)
                if matched_det is not None:
                    det_box = matched_det["box_xywh"]
                    last_det_box = det_box[:]
                    det_iou = box_iou(tracker_box, det_box)
                    det_area_ratio = area_ratio(tracker_box, det_box)
                    force_hard = False
                    if det_iou < args.reinit_iou_threshold:
                        force_hard = True
                    if det_area_ratio > args.size_ratio_threshold:
                        force_hard = True
                    if not math.isnan(tracker_score) and tracker_score < args.low_score_threshold:
                        force_hard = True

                    if force_hard:
                        final_box = det_box[:]
                        tracker.initialize(frame_rgb, {"init_bbox": list(map(float, final_box))})
                        prev_output = {"target_bbox": final_box[:]}
                        hard_reinits += 1
                        mode = "hard_reinit"
                    else:
                        final_box = blend_boxes(tracker_box, det_box, args.detector_weight)
                        tracker.initialize(frame_rgb, {"init_bbox": list(map(float, final_box))})
                        prev_output = {"target_bbox": final_box[:]}
                        soft_corrections += 1
                        mode = "soft_fuse"

            overlay = frame_bgr.copy()
            total_elapsed = total_track_time + total_detect_time
            fps_value = ((frame_index + 1) / total_elapsed) if total_elapsed > 0 else 0.0
            draw_box(overlay, final_box, (0, 255, 0))
            draw_header(overlay, frame_index, fps_value)
            writer.write(overlay)

            if matched_det is None:
                det_values = [float("nan")] * 4
                det_conf = float("nan")
            else:
                det_values = matched_det["box_xywh"]
                det_conf = matched_det["conf"]

            writer_csv.writerow([
                frame_index,
                tracker_box[0], tracker_box[1], tracker_box[2], tracker_box[3],
                final_box[0], final_box[1], final_box[2], final_box[3],
                det_values[0], det_values[1], det_values[2], det_values[3],
                tracker_score, det_conf, mode,
                track_elapsed, detect_elapsed,
            ])

    cap.release()
    writer.release()

    total_elapsed = total_track_time + total_detect_time
    overall_fps = (frame_index + 1) / total_elapsed if total_elapsed > 0 else float("nan")
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("video_path=%s\n" % video_path)
        f.write("detector_model=%s\n" % model_path)
        f.write("detector_backend=%s\n" % detector["backend"])
        if detector["backend"] == "torchscript":
            f.write("device=%s\n" % detector["device"])
        f.write("frames_processed=%d\n" % (frame_index + 1))
        f.write("reported_frame_count=%d\n" % frame_count_reported)
        f.write("video_fps=%.6f\n" % video_fps)
        f.write("frame_width=%d\n" % frame_width)
        f.write("frame_height=%d\n" % frame_height)
        f.write("tracker=%s\n" % args.tracker_name)
        f.write("param=%s\n" % args.param)
        f.write("input_size=%d\n" % args.input_size)
        f.write("detector_conf=%.6f\n" % args.detector_conf)
        f.write("detect_interval=%d\n" % args.detect_interval)
        f.write("low_score_threshold=%.6f\n" % args.low_score_threshold)
        f.write("detection_calls=%d\n" % detection_calls)
        f.write("soft_corrections=%d\n" % soft_corrections)
        f.write("hard_reinits=%d\n" % hard_reinits)
        f.write("overall_fps=%.6f\n" % overall_fps)
        f.write("max_frames=%d\n" % args.max_frames)

    print("video_path=%s" % video_path)
    print("frames_processed=%d" % (frame_index + 1))
    print("detection_calls=%d" % detection_calls)
    print("soft_corrections=%d" % soft_corrections)
    print("hard_reinits=%d" % hard_reinits)
    print("overall_fps=%.6f" % overall_fps)
    print("output_dir=%s" % output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
