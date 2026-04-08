import argparse
import csv
import gc
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTRACKING_ROOT = REPO_ROOT / "MyECOTracker" / "pytracking"

for root in (REPO_ROOT, PYTRACKING_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from pytracking.analysis.extract_results import extract_results
from pytracking.evaluation import Tracker, get_dataset
from pytracking.evaluation.running import run_dataset


def _get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    return auc_curve, auc


def _get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]
    return prec_curve, prec_score


def _load_sequence_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _filter_dataset(dataset, sequence_names: Optional[List[str]]):
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


def _read_time_stats(results_dir: Path, seq_name: str) -> Tuple[int, float, float]:
    time_path = results_dir / f"{seq_name}_time.txt"
    times = np.loadtxt(time_path, delimiter="\t")
    times = np.atleast_1d(times).astype(np.float64)
    total_time = float(np.sum(times))
    frames = int(times.shape[0])
    fps = float(frames / total_time) if total_time > 0 else float("nan")
    return frames, total_time, fps


def _compute_summary(dataset, tracker: Tracker, report_name: str) -> Dict[str, Union[float, int, str]]:
    eval_data = extract_results([tracker], dataset, report_name, skip_missing_seq=False, verbose=False)
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    auc_curve, auc = _get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
    _, precision = _get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

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


def _write_summary_csv(path: Path, summary: Dict[str, Union[float, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _warmup_tracker(dataset, tracker_name: str, parameter_name: str, warmup_frames: int) -> None:
    if warmup_frames <= 0 or not dataset:
        return

    tracker_info = Tracker(tracker_name, parameter_name, None, "warmup")
    params = tracker_info.get_parameters()
    params.visualization = False
    params.debug = 0
    tracker = tracker_info.create_tracker(params)

    seq = dataset[0]
    image = tracker_info._read_image(seq.frames[0])
    prev_output = tracker.initialize(image, seq.init_info()) or {}

    max_frame = min(len(seq.frames), warmup_frames + 1)
    for frame_num in range(1, max_frame):
        image = tracker_info._read_image(seq.frames[frame_num])
        info = seq.frame_info(frame_num)
        info["previous_output"] = prev_output
        prev_output = tracker.track(image, info)

    del tracker
    del prev_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and summarize a pytracking tracker on OTB.")
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--display-name")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=0)
    args = parser.parse_args()

    dataset = list(get_dataset("otb"))
    dataset = _filter_dataset(dataset, _load_sequence_names(args.sequence_file))
    tracker = Tracker(args.tracker_name, args.parameter_name, args.run_id, args.display_name)

    if not args.skip_run:
        _warmup_tracker(dataset, args.tracker_name, args.parameter_name, args.warmup_frames)
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
