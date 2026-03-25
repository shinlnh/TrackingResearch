from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_pytracking_root(repo_root: Path) -> Path:
    candidates = (
        repo_root / "OtherTracker" / "pytracking",
        repo_root / "MyECOTracker" / "pytracking",
        repo_root / "MyTomPTracker" / "pytracking",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a pytracking workspace under OtherTracker/, MyECOTracker/, or MyTomPTracker/."
    )


PYTRACKING_ROOT = _resolve_pytracking_root(REPO_ROOT)

for root in (REPO_ROOT, PYTRACKING_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from OtherTracker.tools.otb_sequences import OTBSequence, load_otb_sequences


@dataclass(frozen=True)
class BenchmarkRow:
    sequence: str
    frames: int
    fps: float
    total_time_sec: float


def _slugify(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        cleaned.append(ch if ch.isalnum() else "_")
    return "".join(cleaned).strip("_")


def _csv_base_name(tracker_name: str, parameter: str | None) -> str:
    parts = [_slugify(tracker_name)]
    if parameter:
        parts.append(_slugify(parameter))
    return "_".join(p for p in parts if p)


def _resolve_opencv_factory(tracker_name: str):
    normalized = tracker_name.strip().lower()
    candidates = {
        "boosting": ("legacy.TrackerBoosting_create", "TrackerBoosting_create"),
        "csrt": ("legacy.TrackerCSRT_create", "TrackerCSRT_create"),
        "kcf": ("legacy.TrackerKCF_create", "TrackerKCF_create"),
        "medianflow": ("legacy.TrackerMedianFlow_create", "TrackerMedianFlow_create"),
        "mil": ("legacy.TrackerMIL_create", "TrackerMIL_create"),
        "mosse": ("legacy.TrackerMOSSE_create", "TrackerMOSSE_create"),
        "tld": ("legacy.TrackerTLD_create", "TrackerTLD_create"),
    }

    if normalized not in candidates:
        available = ", ".join(sorted(candidates))
        raise ValueError(f"Unsupported OpenCV tracker '{tracker_name}'. Available: {available}")

    for path in candidates[normalized]:
        target = cv2
        for attr in path.split("."):
            if not hasattr(target, attr):
                target = None
                break
            target = getattr(target, attr)
        if callable(target):
            return target

    raise RuntimeError(
        f"OpenCV tracker '{tracker_name}' is unavailable in this cv2 build. "
        "Install opencv-contrib-python if needed."
    )


def _run_opencv_sequence(sequence: OTBSequence, tracker_name: str) -> BenchmarkRow:
    frames = sequence.frame_paths()
    init_image = cv2.imread(str(frames[0]))
    if init_image is None:
        raise FileNotFoundError(frames[0])

    tracker = _resolve_opencv_factory(tracker_name)()
    start = time.perf_counter()
    tracker.init(init_image, tuple(sequence.init_rect))

    processed = 1
    for frame_path in frames[1:]:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        tracker.update(image)
        processed += 1

    total_time = time.perf_counter() - start
    fps = processed / total_time if total_time > 0 else 0.0
    return BenchmarkRow(sequence=sequence.name, frames=processed, fps=fps, total_time_sec=total_time)


def _build_pytracking_sequence(sequence: OTBSequence):
    import numpy as np
    from pytracking.evaluation.data import Sequence as PyTrackingSequence

    return PyTrackingSequence(
        name=sequence.name,
        frames=[str(frame_path) for frame_path in sequence.frame_paths()],
        dataset="otb",
        ground_truth_rect=np.asarray(sequence.groundtruth_rects, dtype=np.float64),
    )


def _extract_pytracking_time(output: dict) -> tuple[int, float]:
    time_entries = output.get("time")
    if not time_entries:
        raise RuntimeError("pytracking tracker did not return frame timing data")

    if isinstance(time_entries[0], dict):
        total_time = sum(sum(float(value) for value in frame.values()) for frame in time_entries)
    else:
        total_time = sum(float(value) for value in time_entries)
    return len(time_entries), total_time


def _run_pytracking_sequence(
    sequence: OTBSequence,
    tracker_name: str,
    parameter: str,
    run_id: int | None,
) -> BenchmarkRow:
    from pytracking.evaluation import Tracker as PyTrackingTracker

    tracker = PyTrackingTracker(tracker_name, parameter, run_id)
    pytracking_sequence = _build_pytracking_sequence(sequence)
    output = tracker.run_sequence(
        pytracking_sequence,
        visualization=False,
        debug=0,
        visdom_info={"use_visdom": False},
    )

    frames, total_time = _extract_pytracking_time(output)
    fps = frames / total_time if total_time > 0 else 0.0
    return BenchmarkRow(sequence=sequence.name, frames=frames, fps=fps, total_time_sec=total_time)


def _filter_sequences(
    sequences: list[OTBSequence],
    selected_names: list[str] | None,
    limit: int | None,
) -> list[OTBSequence]:
    if selected_names:
        lookup = {name.lower(): name for name in selected_names}
        filtered = [seq for seq in sequences if seq.name.lower() in lookup]
        missing = sorted(name for name in selected_names if name.lower() not in {seq.name.lower() for seq in filtered})
        if missing:
            raise ValueError(f"Unknown OTB sequences: {', '.join(missing)}")
        sequences = filtered

    if limit is not None:
        sequences = sequences[:limit]

    if not sequences:
        raise ValueError("No sequences selected for benchmarking")
    return sequences


def _summarize_rows(
    rows: list[BenchmarkRow],
    display_name: str,
    backend: str,
    parameter: str | None,
) -> dict[str, float | int | str]:
    total_frames = sum(row.frames for row in rows)
    total_time = sum(row.total_time_sec for row in rows)
    fps_values = [row.fps for row in rows]
    fps_avg_seq = statistics.mean(fps_values)
    fps_weighted_by_frames = total_frames / total_time if total_time > 0 else 0.0

    return {
        "tracker": display_name,
        "backend": backend,
        "parameter": "" if parameter is None else parameter,
        "valid_sequences": len(rows),
        "fps_avg_seq": fps_avg_seq,
        "fps_median_seq": statistics.median(fps_values),
        "fps_global": fps_avg_seq,
        "fps_weighted_by_frames": fps_weighted_by_frames,
        "total_frames": total_frames,
        "total_time_sec": total_time,
    }


def _write_per_sequence_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    "fps": row.fps,
                    "total_time_sec": row.total_time_sec,
                }
            )


def _write_summary_csv(path: Path, summary: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _run_tracker_on_sequence(
    sequence: OTBSequence,
    backend: str,
    tracker_name: str,
    parameter: str | None,
    run_id: int | None,
) -> BenchmarkRow:
    if backend == "opencv":
        return _run_opencv_sequence(sequence, tracker_name)
    if backend == "pytracking":
        if not parameter:
            raise ValueError("--parameter is required for pytracking trackers")
        return _run_pytracking_sequence(sequence, tracker_name, parameter, run_id)
    raise ValueError(f"Unsupported backend: {backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a Python tracker on OTB and emit a single FPS_global summary."
    )
    parser.add_argument("--backend", choices=("opencv", "pytracking"), required=True)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--display-name")
    parser.add_argument("--otb-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    display_name = args.display_name or args.tracker_name.upper()
    sequences = load_otb_sequences(args.otb_root, args.sequence_file)
    sequences = _filter_sequences(sequences, args.sequences, args.limit)

    rows: list[BenchmarkRow] = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"[run ] {index}/{len(sequences)} {sequence.name} -> {display_name}")
        row = _run_tracker_on_sequence(
            sequence=sequence,
            backend=args.backend,
            tracker_name=args.tracker_name,
            parameter=args.parameter,
            run_id=args.run_id,
        )
        rows.append(row)
        print(f"[done] {sequence.name}: fps={row.fps:.6f}, time={row.total_time_sec:.6f}s")

    summary = _summarize_rows(rows, display_name, args.backend, args.parameter)
    base_name = _csv_base_name(args.tracker_name, args.parameter)
    _write_per_sequence_csv(args.out_dir / f"{base_name}_per_sequence.csv", rows)
    _write_summary_csv(args.out_dir / f"{base_name}_summary.csv", summary)

    print(f"FPS_global={float(summary['fps_global']):.6f}")
    print(f"Wrote benchmark outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
