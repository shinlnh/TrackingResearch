from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OtherTracker.tools.fps_benchmark import _resolve_opencv_factory


def _bootstrap_pytracking(pytracking_root: Path):
    root_str = str(pytracking_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from pytracking.analysis.plot_results import check_and_load_precomputed_results, get_auc_curve, get_prec_curve
    from pytracking.evaluation import get_dataset

    return check_and_load_precomputed_results, get_auc_curve, get_prec_curve, get_dataset


def _scope_name(sequence_file: Path | None) -> str:
    if sequence_file is None:
        return "testset280"
    stem = sequence_file.stem
    if stem.endswith("_sequences"):
        stem = stem[: -len("_sequences")]
    return stem


def _load_dataset(get_dataset, sequence_file: Path | None, sequence_name: str | None, limit: int | None):
    dataset = list(get_dataset("lasot"))
    by_name = {seq.name: seq for seq in dataset}

    if sequence_file is not None:
        raw_names = [
            line.strip()
            for line in sequence_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing = [name for name in raw_names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown LaSOT sequences: {', '.join(missing)}")
        dataset = [by_name[name] for name in raw_names]

    if sequence_name is not None:
        if sequence_name not in by_name:
            raise ValueError(f"Unknown LaSOT sequence: {sequence_name}")
        dataset = [by_name[sequence_name]]

    if limit is not None:
        dataset = dataset[:limit]

    if not dataset:
        raise ValueError("No sequences selected for benchmarking")

    return dataset


def _write_sequence_outputs(results_dir: Path, sequence_name: str, boxes: list[tuple[float, float, float, float]], times: list[float]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(results_dir / f"{sequence_name}.txt", np.asarray(boxes, dtype=np.float64), fmt="%.6f", delimiter=",")
    np.savetxt(results_dir / f"{sequence_name}_time.txt", np.asarray(times, dtype=np.float64), fmt="%.9f")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_sequence(seq, tracker_name: str):
    frames = [Path(frame) for frame in seq.frames]
    tracker = _resolve_opencv_factory(tracker_name)()

    init_image = cv2.imread(str(frames[0]))
    if init_image is None:
        raise FileNotFoundError(frames[0])

    boxes: list[tuple[float, float, float, float]] = [tuple(float(v) for v in seq.ground_truth_rect[0].tolist())]
    times: list[float] = []

    start = time.perf_counter()
    tracker.init(init_image, tuple(float(v) for v in seq.ground_truth_rect[0].tolist()))
    times.append(time.perf_counter() - start)

    for frame_path in frames[1:]:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        start = time.perf_counter()
        ok, bbox = tracker.update(image)
        times.append(time.perf_counter() - start)

        if ok:
            boxes.append(tuple(float(v) for v in bbox))
        else:
            boxes.append(boxes[-1])

    total_time = float(sum(times))
    fps = float(len(frames) / total_time) if total_time > 0 else 0.0
    return boxes, times, fps, total_time, len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenCV CSRT on LaSOT and save results under OtherTracker/lasot.")
    parser.add_argument("--sequence-file", type=Path, help="Optional file listing LaSOT sequence names to run.")
    parser.add_argument("--sequence", help="Optional single LaSOT sequence to run.")
    parser.add_argument("--limit", type=int, help="Optional limit after filtering.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults under OtherTracker/lasot/lasot936/CSRT.")
    parser.add_argument("--display-name", default="CSRT")
    parser.add_argument("--tracker-name", default="csrt")
    args = parser.parse_args(argv)

    pytracking_root = REPO_ROOT / "OtherTracker" / "pytracking"
    check_and_load_precomputed_results, get_auc_curve, get_prec_curve, get_dataset = _bootstrap_pytracking(pytracking_root)

    dataset = _load_dataset(get_dataset, args.sequence_file, args.sequence, args.limit)
    scope = _scope_name(args.sequence_file)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "CSRT"
    output_dir = output_dir.resolve()
    results_dir = output_dir / "tracking_results" / args.display_name
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    fps_values: list[float] = []
    total_frames = 0
    total_time = 0.0

    for index, seq in enumerate(dataset, start=1):
        print(f"[run ] {index:2d}/{len(dataset):2d} {seq.name} -> {args.display_name}", flush=True)
        boxes, times, fps, seq_time, frames = _run_sequence(seq, args.tracker_name)
        _write_sequence_outputs(results_dir, seq.name, boxes, times)

        rows.append(
            {
                "sequence": seq.name,
                "frames": frames,
                "fps": fps,
                "total_time_sec": seq_time,
                "txt_path": str(results_dir / f"{seq.name}.txt"),
            }
        )
        fps_values.append(fps)
        total_frames += frames
        total_time += seq_time
        print(f"[done] {seq.name}: fps={fps:.6f}, time={seq_time:.6f}s", flush=True)

    _write_csv(manifest_path, ["sequence", "frames", "fps", "total_time_sec", "txt_path"], rows)

    tracker = SimpleNamespace(
        name=args.display_name,
        parameter_name=args.tracker_name,
        run_id=None,
        display_name=args.display_name,
        results_dir=str(results_dir),
    )
    report_name = f"lasot_{scope}_{args.tracker_name}_othertracker"
    eval_data = check_and_load_precomputed_results(
        [tracker], dataset, report_name, force_evaluation=True, skip_missing_seq=False, verbose=False
    )
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    auc_curve, auc = get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
    _, precision = get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

    summary = {
        "tracker": args.display_name,
        "backend": "opencv",
        "parameter": args.tracker_name,
        "scope": scope,
        "valid_sequences": int(valid.sum().item()),
        "AUC": float(auc[0]),
        "Precision": float(precision[0]),
        "Success50": float(auc_curve[0, 10]),
        "FPS_avg_seq": float(statistics.mean(fps_values)),
        "FPS_median_seq": float(statistics.median(fps_values)),
        "FPS_weighted_by_frames": float(total_frames / total_time) if total_time > 0 else float("nan"),
        "total_frames": int(total_frames),
        "total_time_sec": float(total_time),
    }
    _write_csv(summary_path, list(summary.keys()), [summary])

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
