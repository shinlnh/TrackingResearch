from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OtherTracker.tools.otb_sequences import OTBSequence, load_otb_sequences
from OtherTracker.tools.fps_benchmark import _resolve_opencv_factory


@dataclass(frozen=True)
class SequenceRun:
    sequence: str
    frames: int
    fps: float
    total_time_sec: float


def _write_sequence_outputs(
    out_dir: Path,
    tracker_label: str,
    sequence: OTBSequence,
    boxes: list[tuple[float, float, float, float]],
    times: list[float],
) -> None:
    txt_dir = out_dir / "txt_results" / tracker_label
    txt_dir.mkdir(parents=True, exist_ok=True)

    boxes_path = txt_dir / f"{sequence.name}.txt"
    times_path = txt_dir / f"{sequence.name}_time.txt"

    np.savetxt(boxes_path, np.asarray(boxes, dtype=np.float64), fmt="%.6f", delimiter=",")
    np.savetxt(times_path, np.asarray(times, dtype=np.float64), fmt="%.9f")


def _run_sequence(sequence: OTBSequence, tracker_name: str) -> tuple[list[tuple[float, float, float, float]], list[float], SequenceRun]:
    frames = sequence.frame_paths()
    tracker = _resolve_opencv_factory(tracker_name)()

    init_image = cv2.imread(str(frames[0]))
    if init_image is None:
        raise FileNotFoundError(frames[0])

    boxes: list[tuple[float, float, float, float]] = [tuple(float(v) for v in sequence.init_rect)]
    times: list[float] = []

    start = time.perf_counter()
    tracker.init(init_image, tuple(sequence.init_rect))
    times.append(time.perf_counter() - start)

    for frame_path in frames[1:]:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        start = time.perf_counter()
        ok, bbox = tracker.update(image)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if ok:
            boxes.append(tuple(float(v) for v in bbox))
        else:
            boxes.append(boxes[-1])

    total_time = float(sum(times))
    fps = float(len(frames) / total_time) if total_time > 0 else 0.0
    summary = SequenceRun(
        sequence=sequence.name,
        frames=len(frames),
        fps=fps,
        total_time_sec=total_time,
    )
    return boxes, times, summary


def _write_manifest(path: Path, rows: list[SequenceRun]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sequence", "frames", "fps", "total_time_sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "frames": row.frames,
                    "fps": f"{row.fps:.6f}",
                    "total_time_sec": f"{row.total_time_sec:.9f}",
                }
            )


def _write_summary(path: Path, rows: list[SequenceRun], tracker_label: str, tracker_name: str) -> None:
    total_frames = sum(row.frames for row in rows)
    total_time = sum(row.total_time_sec for row in rows)
    fps_values = [row.fps for row in rows]
    summary = {
        "tracker": tracker_label,
        "backend": "opencv",
        "parameter": tracker_name,
        "valid_sequences": len(rows),
        "fps_avg_seq": statistics.mean(fps_values),
        "fps_median_seq": statistics.median(fps_values),
        "fps_global": statistics.mean(fps_values),
        "fps_weighted_by_frames": (total_frames / total_time) if total_time > 0 else 0.0,
        "total_frames": total_frames,
        "total_time_sec": total_time,
    }
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tracker",
                "backend",
                "parameter",
                "valid_sequences",
                "fps_avg_seq",
                "fps_median_seq",
                "fps_global",
                "fps_weighted_by_frames",
                "total_frames",
                "total_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an OpenCV tracker on OTB and save boxes/time files.")
    parser.add_argument("--tracker-name", required=True, help="OpenCV tracker factory name, e.g. csrt or kcf.")
    parser.add_argument("--display-name", required=True, help="Tracker label used in saved outputs.")
    parser.add_argument("--otb-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    sequences = load_otb_sequences(args.otb_root, args.sequence_file)
    if args.sequences:
        wanted = {name.lower() for name in args.sequences}
        sequences = [seq for seq in sequences if seq.name.lower() in wanted]
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise ValueError("No sequences selected for tracking.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[SequenceRun] = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"[run ] {index}/{len(sequences)} {sequence.name} -> {args.display_name}")
        boxes, times, row = _run_sequence(sequence, args.tracker_name)
        _write_sequence_outputs(args.out_dir, args.display_name, sequence, boxes, times)
        rows.append(row)
        print(f"[done] {sequence.name}: fps={row.fps:.6f}, time={row.total_time_sec:.6f}s")

    _write_manifest(args.out_dir / "manifest.csv", rows)
    _write_summary(args.out_dir / "summary.csv", rows, args.display_name, args.tracker_name)
    print(f"Wrote tracking outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
