import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from pytracking.analysis.extract_results import extract_results
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation import Tracker, get_dataset
from pytracking.evaluation.lasotdataset import LaSOTDataset


SUCCESS50_INDEX = 10
SUCCESS75_INDEX = 15
PRECISION_INDEX = 20
NORM_PRECISION_INDEX = 20


def load_sequence_names(path):
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def filter_dataset(dataset, sequence_names):
    if not sequence_names:
        return list(dataset)

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
        raise ValueError("Unknown sequences for dataset filter: {}".format(", ".join(missing)))

    return filtered


def load_dataset(dataset_name, sequence_names):
    if dataset_name == "lasot" and sequence_names:
        return LaSOTDataset(sequence_list=sequence_names).get_sequence_list()
    return filter_dataset(get_dataset(dataset_name), sequence_names)


def read_time_stats(tracker_results_dir, seq_name):
    time_path = tracker_results_dir / "{}_time.txt".format(seq_name)
    if not time_path.exists():
        return float("nan"), 0, float("nan")

    times = np.loadtxt(str(time_path), delimiter="\t")
    times = np.atleast_1d(times).astype(np.float64)
    total_time = float(np.sum(times))
    frames = int(times.shape[0])
    fps = float(frames / total_time) if total_time > 0 else float("nan")
    return fps, frames, total_time


def build_rows(dataset, eval_data, tracker_results_dir):
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    avg_overlap_all = torch.tensor(eval_data["avg_overlap_all"], dtype=torch.float64)
    success_overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float64)
    precision_curve = torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float64)
    norm_precision_curve = torch.tensor(eval_data["ave_success_rate_plot_center_norm"], dtype=torch.float64)

    rows = []
    missing_sequences = []

    for seq_id, seq in enumerate(dataset):
        if not valid[seq_id]:
            missing_sequences.append(seq.name)
            continue

        fps, frames, total_time = read_time_stats(tracker_results_dir, seq.name)
        rows.append(
            {
                "sequence_index": len(rows) + 1,
                "sequence": seq.name,
                "AUC": float(avg_overlap_all[seq_id, 0] * 100.0),
                "Precision": float(precision_curve[seq_id, 0, PRECISION_INDEX] * 100.0),
                "Norm_Precision": float(norm_precision_curve[seq_id, 0, NORM_PRECISION_INDEX] * 100.0),
                "Success50": float(success_overlap[seq_id, 0, SUCCESS50_INDEX] * 100.0),
                "Success75": float(success_overlap[seq_id, 0, SUCCESS75_INDEX] * 100.0),
                "FPS": fps,
                "frames": frames,
                "total_time_sec": total_time,
            }
        )

    return rows, missing_sequences


def compute_summary(rows, eval_data, tracker_name, parameter_name, run_id, dataset_label, missing_sequences):
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    success_overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float64)[valid].mean(0) * 100.0
    precision_curve = torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float64)[valid].mean(0) * 100.0
    norm_precision_curve = (
        torch.tensor(eval_data["ave_success_rate_plot_center_norm"], dtype=torch.float64)[valid].mean(0) * 100.0
    )

    auc_values = torch.tensor([row["AUC"] for row in rows], dtype=torch.float64)
    fps_values = np.array([row["FPS"] for row in rows], dtype=np.float64)
    total_frames = int(sum(row["frames"] for row in rows))
    total_time = float(sum(row["total_time_sec"] for row in rows))

    summary = {
        "tracker": "{}_{}_{}".format(tracker_name, parameter_name, run_id),
        "dataset": dataset_label,
        "valid_sequences": "{}/{}".format(len(rows), len(rows) + len(missing_sequences)),
        "missing_sequences": ", ".join(missing_sequences),
        "AUC_mean": float(auc_values.mean()) if len(rows) > 0 else float("nan"),
        "Precision_mean": float(precision_curve[0, PRECISION_INDEX]),
        "Norm_Precision_mean": float(norm_precision_curve[0, NORM_PRECISION_INDEX]),
        "Success50_mean": float(success_overlap[0, SUCCESS50_INDEX]),
        "Success75_mean": float(success_overlap[0, SUCCESS75_INDEX]),
        "FPS_avg_seq": float(np.nanmean(fps_values)) if len(rows) > 0 else float("nan"),
        "FPS_median_seq": float(np.nanmedian(fps_values)) if len(rows) > 0 else float("nan"),
        "FPS_weighted_by_frames": float(total_frames / total_time) if total_time > 0 else float("nan"),
        "total_frames": total_frames,
        "total_time_sec": total_time,
    }
    return summary


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_eval_data(tracker, dataset, report_name):
    settings = env_settings()
    report_dir = Path(settings.result_plot_path) / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    return extract_results([tracker], dataset, report_name, skip_missing_seq=False, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Export per-sequence tracking metrics to CSV.")
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--dataset-name", default="otb")
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--report-name", default="")
    args = parser.parse_args()

    sequence_names = load_sequence_names(args.sequence_file)
    dataset = load_dataset(args.dataset_name, sequence_names)
    tracker = Tracker(args.tracker_name, args.parameter_name, args.run_id)
    dataset_label = args.dataset_label if args.dataset_label else (
        args.sequence_file.stem if args.sequence_file is not None else args.dataset_name
    )
    report_name = (
        args.report_name
        if args.report_name
        else "{}_{}_{}_{}".format(dataset_label, args.tracker_name, args.parameter_name, args.run_id)
    )

    eval_data = compute_eval_data(tracker, dataset, report_name)

    rows, missing_sequences = build_rows(dataset, eval_data, Path(tracker.results_dir))
    summary = compute_summary(
        rows,
        eval_data,
        args.tracker_name,
        args.parameter_name,
        args.run_id,
        dataset_label,
        missing_sequences,
    )

    row_fields = [
        "sequence_index",
        "sequence",
        "AUC",
        "Precision",
        "Norm_Precision",
        "Success50",
        "Success75",
        "FPS",
        "frames",
        "total_time_sec",
    ]
    summary_fields = [
        "tracker",
        "dataset",
        "valid_sequences",
        "missing_sequences",
        "AUC_mean",
        "Precision_mean",
        "Norm_Precision_mean",
        "Success50_mean",
        "Success75_mean",
        "FPS_avg_seq",
        "FPS_median_seq",
        "FPS_weighted_by_frames",
        "total_frames",
        "total_time_sec",
    ]

    write_csv(args.rows_csv, row_fields, rows)
    write_csv(args.summary_csv, summary_fields, [summary])

    print("rows_csv={}".format(args.rows_csv))
    print("summary_csv={}".format(args.summary_csv))
    print(
        "AUC={:.4f} FPS_avg_seq={:.4f} FPS_weighted={:.4f}".format(
            summary["AUC_mean"], summary["FPS_avg_seq"], summary["FPS_weighted_by_frames"]
        )
    )


if __name__ == "__main__":
    main()
