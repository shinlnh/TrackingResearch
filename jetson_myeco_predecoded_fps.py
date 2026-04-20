#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Decode video to frames, then run pure MyECO on the decoded sequence.")
    parser.add_argument("video_path", type=Path, help="Input video path.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory to store decoded frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and prediction csv.")
    parser.add_argument("--init-xywh", nargs=4, type=float, required=True, metavar=("X", "Y", "W", "H"), help="Initial bbox in XYWH pixels.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936", help="PyTracking parameter name.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Frame image extension.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality if ext=jpg.")
    parser.add_argument("--reuse-frames", action="store_true", help="Reuse existing decoded frames if present.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit. 0 means full sequence.")
    return parser.parse_args()


def resolve_project_root(script_dir):
    for candidate in [script_dir] + list(script_dir.parents):
        if (candidate / "pytracking").is_dir():
            return candidate
        if (candidate / "MyECOTracker" / "pytracking").is_dir():
            return candidate / "MyECOTracker"
    raise RuntimeError("Could not locate pytracking project root from %s" % script_dir)


def setup_pytracking(project_root):
    pytracking_dir = project_root / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def decode_video(video_path, frames_dir, ext, quality, reuse_frames, max_frames):
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = "*." + ext
    existing = sorted(frames_dir.glob(pattern))
    if reuse_frames and existing:
        return existing

    for old in existing:
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: %s" % video_path)

    frame_paths = []
    frame_index = 0
    params = []
    if ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    while True:
        if max_frames > 0 and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_path = frames_dir / ("frame_%06d.%s" % (frame_index, ext))
        if params:
            cv2.imwrite(str(frame_path), frame, params)
        else:
            cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_index += 1

    cap.release()
    return frame_paths


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
    frames_dir = args.frames_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decode_start = time.perf_counter()
    frame_paths = decode_video(video_path, frames_dir, args.ext, args.quality, args.reuse_frames, args.max_frames)
    decode_elapsed = time.perf_counter() - decode_start
    if not frame_paths:
        raise RuntimeError("No frames decoded from %s" % video_path)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError("Failed to read decoded frame %s" % frame_paths[0])

    tracker = create_tracker(project_root, args.tracker_name, args.param)
    init_bbox = [float(v) for v in args.init_xywh]

    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    out = tracker.initialize(first_rgb, {"init_bbox": init_bbox}) or {}
    init_elapsed = time.perf_counter() - start
    prev_output = dict(out)

    predictions = [[float(v) for v in out.get("target_bbox", init_bbox)]]
    times_s = [init_elapsed]

    for frame_path in frame_paths[1:]:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError("Failed to read %s" % frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        out = tracker.track(image_rgb, {"previous_output": prev_output}) or {}
        elapsed = time.perf_counter() - start
        prev_output = dict(out)
        predictions.append([float(v) for v in out["target_bbox"]])
        times_s.append(elapsed)

    track_total = sum(times_s)
    total_frames = len(frame_paths)
    fps_including_init = (float(total_frames) / track_total) if track_total > 0 else float("nan")
    track_total_excl_init = sum(times_s[1:])
    fps_excluding_init = (float(max(0, total_frames - 1)) / track_total_excl_init) if track_total_excl_init > 0 else float("nan")

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("video_path=%s\n" % video_path)
        f.write("frames_dir=%s\n" % frames_dir)
        f.write("frames_decoded=%d\n" % total_frames)
        f.write("tracker=%s\n" % args.tracker_name)
        f.write("param=%s\n" % args.param)
        f.write("init_bbox_xywh=%s\n" % ",".join("%.6f" % v for v in init_bbox))
        f.write("decode_time_total_s=%.6f\n" % decode_elapsed)
        f.write("track_time_total_s=%.6f\n" % track_total)
        f.write("track_time_excluding_init_s=%.6f\n" % track_total_excl_init)
        f.write("fps_including_init=%.6f\n" % fps_including_init)
        f.write("fps_excluding_init=%.6f\n" % fps_excluding_init)
        f.write("max_frames=%d\n" % args.max_frames)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "frame_path", "pred_x", "pred_y", "pred_w", "pred_h", "time_s"])
        for idx, (frame_path, pred, elapsed) in enumerate(zip(frame_paths, predictions, times_s)):
            writer.writerow([idx, str(frame_path), pred[0], pred[1], pred[2], pred[3], elapsed])

    print("video_path=%s" % video_path)
    print("frames_decoded=%d" % total_frames)
    print("decode_time_total_s=%.6f" % decode_elapsed)
    print("track_time_total_s=%.6f" % track_total)
    print("fps_including_init=%.6f" % fps_including_init)
    print("fps_excluding_init=%.6f" % fps_excluding_init)
    print("output_dir=%s" % output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
