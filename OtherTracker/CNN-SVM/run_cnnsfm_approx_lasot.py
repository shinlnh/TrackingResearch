from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cnnsfm_approx import track_sequence


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


def _write_boxes(results_dir: Path, sequence: str, boxes: list[tuple[float, float, float, float]], times: list[float]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / f"{sequence}.txt").open("w", encoding="utf-8") as fh:
        for box in boxes:
            fh.write(",".join(f"{value:.6f}" for value in box) + "\n")

    with (results_dir / f"{sequence}_time.txt").open("w", encoding="utf-8") as fh:
        for value in times:
            fh.write(f"{value:.9f}\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CNN-SVM-Approx on LaSOT and save results under OtherTracker/lasot.")
    parser.add_argument("--sequence-file", type=Path, help="Optional file listing LaSOT sequence names to run.")
    parser.add_argument("--sequence", help="Optional single LaSOT sequence to run.")
    parser.add_argument("--limit", type=int, help="Optional limit after filtering.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults under OtherTracker/lasot/lasot936/CNN-SVM.")
    parser.add_argument("--display-name", default="CNN-SVM")
    parser.add_argument("--device", help="Tracking device. Defaults to cuda when available.")
    args = parser.parse_args(argv)

    pytracking_root = REPO_ROOT / "OtherTracker" / "pytracking"
    check_and_load_precomputed_results, get_auc_curve, get_prec_curve, get_dataset = _bootstrap_pytracking(pytracking_root)

    dataset = _load_dataset(get_dataset, args.sequence_file, args.sequence, args.limit)
    scope = _scope_name(args.sequence_file)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "CNN-SVM"
    output_dir = output_dir.resolve()
    results_dir = output_dir / "tracking_results" / args.display_name
    manifest_path = output_dir / "manifest.csv"
    per_sequence_path = output_dir / "cnn_svm_per_sequence.csv"
    summary_path = output_dir / "summary.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    fps_values: list[float] = []
    total_frames = 0
    total_time = 0.0

    for index, seq in enumerate(dataset, start=1):
        print(f"[run ] {index:2d}/{len(dataset):2d} {seq.name} -> CNN-SVM-Approx", flush=True)
        seq_start = time.perf_counter()
        boxes, times = track_sequence(
            [str(path) for path in seq.frames],
            tuple(float(v) for v in seq.ground_truth_rect[0].tolist()),
            device=args.device,
            log_prefix=f"[CNN-SVM] {seq.name}",
        )
        wall_total = time.perf_counter() - seq_start

        times = list(times)
        tracked_time = float(sum(times[1:])) if len(times) > 1 else 0.0
        if times:
            times[0] = max(wall_total - tracked_time, 0.0)

        seq_time = float(sum(times))
        frames = len(seq.frames)
        fps = float(frames / seq_time) if seq_time > 0 else float("nan")

        _write_boxes(results_dir, seq.name, boxes, times)

        manifest_rows.append(
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_csv(
        manifest_path,
        ["sequence", "frames", "fps", "total_time_sec", "txt_path"],
        manifest_rows,
    )

    tracker = SimpleNamespace(
        name=args.display_name,
        parameter_name="approx_alexnet_fc6_saliency",
        run_id=None,
        display_name=args.display_name,
        results_dir=str(results_dir),
    )
    report_name = f"lasot_{scope}_cnn_svm_othertracker"
    eval_data = check_and_load_precomputed_results(
        [tracker], dataset, report_name, force_evaluation=True, skip_missing_seq=False, verbose=False
    )
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    auc_curve, auc = get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
    _, precision = get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

    fps_values_np = np.asarray(fps_values, dtype=np.float64)
    summary = {
        "tracker": args.display_name,
        "parameter": "approx_alexnet_fc6_saliency",
        "scope": scope,
        "valid_sequences": int(valid.sum().item()),
        "AUC": float(auc[0]),
        "Precision": float(precision[0]),
        "Success50": float(auc_curve[0, 10]),
        "FPS_avg_seq": float(np.nanmean(fps_values_np)),
        "FPS_median_seq": float(np.nanmedian(fps_values_np)),
        "FPS_weighted_by_frames": float(total_frames / total_time) if total_time > 0 else float("nan"),
        "total_frames": int(total_frames),
        "total_time_sec": float(total_time),
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }
    _write_csv(summary_path, list(summary.keys()), [summary])
    _write_csv(per_sequence_path, ["sequence", "frames", "fps", "total_time_sec", "txt_path"], manifest_rows)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
