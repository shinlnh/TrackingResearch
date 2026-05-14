import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def load_per_sequence_rows(path: Path):
    rows = read_csv_rows(path)
    parsed = []
    for row in rows:
        parsed.append(
            {
                "sequence_index": int(row["sequence_index"]),
                "sequence": row["sequence"],
                "AUC": parse_float(row["AUC"]),
                "FPS": parse_float(row["FPS"]),
            }
        )
    return parsed


def load_summary(path: Path | None):
    if path is None or not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    row = rows[0]
    return {
        "tracker": row.get("tracker", ""),
        "dataset": row.get("dataset", ""),
        "AUC_mean": parse_float(row.get("AUC_mean", "")),
        "FPS_avg_seq": parse_float(row.get("FPS_avg_seq", "")),
        "FPS_weighted_by_frames": parse_float(row.get("FPS_weighted_by_frames", "")),
    }


def load_eval_data(path: Path | None):
    if path is None or not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def format_tracker_label(summary):
    if summary is None:
        return "tracker"
    tracker = summary.get("tracker", "")
    if not tracker:
        return "tracker"
    return tracker.replace("eco_", "")


def format_dataset_label(summary):
    if summary is None:
        return "LaSOT"

    dataset = summary.get("dataset", "")
    if not dataset:
        return "LaSOT"

    dataset = dataset.replace("_headtail40", "")
    dataset = dataset.replace("HeadTail40", "")
    dataset = dataset.replace("_", " ").strip()

    if dataset.lower() == "lasot":
        return "LaSOT"

    return dataset


def save_fps_plot(path: Path, rows, summary):
    indices = np.array([row["sequence_index"] for row in rows], dtype=np.int32)
    fps = np.array([row["FPS"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.plot(indices, fps, color="#4f79b6", linewidth=2.2)
    ax.set_title("FPS JetSon on LaSOT", fontsize=18, pad=12)
    ax.set_xlabel("Sequence Index", fontsize=14)
    ax.set_ylabel("FPS", fontsize=14)
    ax.grid(True, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.35)
    ax.set_xlim(1, len(indices))
    ymax = max(30.0, float(np.nanmax(fps)) + 3.0)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=12)

    if summary is not None and np.isfinite(summary["FPS_avg_seq"]):
        avg_text = "[FPS_avg : {:.2f}]".format(summary["FPS_avg_seq"])
        ax.text(
            0.98,
            0.93,
            avg_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            color="#4f79b6",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#4f79b6", linewidth=1.2),
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_success_plot(path: Path, summary, eval_data):
    if eval_data is None:
        raise ValueError("eval_data.pkl is required to draw the success plot")

    thresholds = np.array(eval_data["threshold_set_overlap"], dtype=np.float64)
    valid = np.array(eval_data["valid_sequence"], dtype=np.int32).astype(bool)
    success = np.array(eval_data["ave_success_rate_plot_overlap"], dtype=np.float64)
    success_curve = success[valid].mean(axis=0)[0] * 100.0
    auc_value = summary["AUC_mean"] if summary is not None else float(success_curve.mean())

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(
        thresholds,
        success_curve,
        color="#187d63",
        linewidth=2.5,
        label="AUC = {:.2f}".format(auc_value),
    )
    ax.fill_between(thresholds, success_curve, 0.0, color="#187d63", alpha=0.12)
    ax.set_title("AUC JetSon on LaSOT", fontsize=18, pad=12)
    ax.set_xlabel("Overlap threshold", fontsize=14)
    ax.set_ylabel("Overlap Precision [%]", fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AUC and FPS charts from per-sequence CSV metrics.")
    parser.add_argument("--per-sequence-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--eval-data-pkl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    rows = load_per_sequence_rows(args.per_sequence_csv)
    summary = load_summary(args.summary_csv)
    eval_data = load_eval_data(args.eval_data_pkl)
    output_dir = args.output_dir if args.output_dir is not None else args.per_sequence_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_path = output_dir / "fps_plot.png"
    success_path = output_dir / "success_plot.png"

    save_fps_plot(fps_path, rows, summary)
    save_success_plot(success_path, summary, eval_data)

    print("fps_png={}".format(fps_path))
    print("success_png={}".format(success_path))


if __name__ == "__main__":
    main()
