from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTRACKING_ROOT = REPO_ROOT / "MyECOTracker" / "pytracking"

for root in (REPO_ROOT, PYTRACKING_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from pytracking.analysis.plot_results import check_and_load_precomputed_results, get_auc_curve, get_prec_curve
from pytracking.evaluation import Tracker, get_dataset
from pytracking.evaluation.running import run_dataset


def _load_sequence_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _filter_dataset(dataset, sequence_names: list[str] | None):
    if not sequence_names:
        return dataset

    by_name = {seq.name: seq for seq in dataset}
    filtered = []
    missing = []
    for name in sequence_names:
        seq = by_name.get(name)
        if seq is None:
            missing.append(name)
        else:
            filtered.append(seq)

    if missing:
        raise ValueError(f"Unknown OTB sequences: {', '.join(missing)}")
    return filtered


def _read_time_stats(results_dir: Path, seq_name: str) -> tuple[int, float, float]:
    time_path = results_dir / f"{seq_name}_time.txt"
    times = np.loadtxt(time_path, delimiter="\t")
    times = np.atleast_1d(times).astype(np.float64)
    total_time = float(np.sum(times))
    frames = int(times.shape[0])
    fps = float(frames / total_time) if total_time > 0 else float("nan")
    return frames, total_time, fps


def _compute_summary(dataset, tracker: Tracker, report_name: str) -> dict[str, float | int | str]:
    eval_data = check_and_load_precomputed_results(
        [tracker], dataset, report_name, force_evaluation=True, skip_missing_seq=False, verbose=False
    )
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    auc_curve, auc = get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
    _, precision = get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

    frames_total = 0
    total_time = 0.0
    fps_values = []
    results_dir = Path(tracker.results_dir)
    for seq in dataset:
        frames, seq_time, seq_fps = _read_time_stats(results_dir, seq.name)
        frames_total += frames
        total_time += seq_time
        fps_values.append(seq_fps)

    fps_values_np = np.asarray(fps_values, dtype=np.float64)
    return {
        "tracker": tracker.display_name or f"{tracker.name}_{tracker.parameter_name}",
        "dataset": "OTB",
        "valid_sequences": int(valid.sum().item()),
        "AUC": float(auc[0]),
        "Precision": float(precision[0]),
        "Success50": float(auc_curve[0, 10]),
        "FPS_avg_seq": float(np.nanmean(fps_values_np)),
        "FPS_median_seq": float(np.nanmedian(fps_values_np)),
        "FPS_weighted_by_frames": float(frames_total / total_time) if total_time > 0 else float("nan"),
        "total_frames": frames_total,
        "total_time_sec": total_time,
    }


def _write_summary_csv(path: Path, summary: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and summarize a pytracking tracker on OTB.")
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--display-name")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    dataset = list(get_dataset("otb"))
    dataset = _filter_dataset(dataset, _load_sequence_names(args.sequence_file))
    tracker = Tracker(args.tracker_name, args.parameter_name, args.run_id, args.display_name)

    if not args.skip_run:
        run_dataset(dataset, [tracker], debug=0, threads=0, visdom_info={"use_visdom": False})

    report_name = f"profile_{args.tracker_name}_{args.parameter_name}_{'full' if args.sequence_file is None else args.sequence_file.stem}"
    summary = _compute_summary(dataset, tracker, report_name)

    if args.summary_csv:
        _write_summary_csv(args.summary_csv, summary)

    print(
        " | ".join(
            [
                f"tracker={summary['tracker']}",
                f"valid={summary['valid_sequences']}",
                f"AUC={summary['AUC']:.4f}",
                f"Precision={summary['Precision']:.4f}",
                f"Success50={summary['Success50']:.4f}",
                f"FPS_avg_seq={summary['FPS_avg_seq']:.4f}",
                f"FPS_weighted={summary['FPS_weighted_by_frames']:.4f}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
