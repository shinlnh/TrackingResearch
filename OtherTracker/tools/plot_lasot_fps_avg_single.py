from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FpsEntry:
    tracker: str
    fps_avg: float


TRACKER_SUMMARY_PATHS = {
    "MDNet": "OtherTracker/lasot/lasot936/MDNet/summary.csv",
    "ECO": "OtherTracker/lasot/lasot936/ECO/summary.csv",
    "DeepSRDCF": "OtherTracker/lasot/lasot936/DeepSRDCF/summary.csv",
    "SRDCFdecon": "OtherTracker/lasot/lasot936/SRDCFdecon/summary.csv",
    "ToMP": "OtherTracker/lasot/lasot936/ToMP/summary.csv",
    "SRDCF": "OtherTracker/lasot/lasot936/SRDCF/summary.csv",
    "Staple": "OtherTracker/lasot/lasot936/Staple/summary.csv",
    "HDT": "OtherTracker/lasot/lasot936/HDT/summary.csv",
    "CF2": "OtherTracker/lasot/lasot936/CF2/summary.csv",
    "LCT": "OtherTracker/lasot/lasot936/LCT/summary.csv",
    "SAMF": "OtherTracker/lasot/lasot936/SAMF/summary.csv",
    "CNN-SVM": "OtherTracker/lasot/lasot936/CNN-SVM/summary.csv",
    "MEEM": "OtherTracker/lasot/lasot936/MEEM/summary.csv",
    "CSRT": "OtherTracker/lasot/lasot936/CSRT/summary.csv",
    "DSST": "OtherTracker/lasot/lasot936/DSST/summary.csv",
    "KCF": "OtherTracker/lasot/lasot936/KCF/summary.csv",
}


def _format_fps(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _load_summary_fps(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return float(row["FPS_avg_seq"])


def _load_mytracker_fps(log_path: Path) -> float:
    fps_avg = None
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = log_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Could not decode log file: {log_path}")

    for line in text.splitlines():
        if line.startswith("FPS_avg_seq:"):
            fps_avg = float(line.split(":", 1)[1].strip())
    if fps_avg is None:
        raise ValueError(f"Could not find FPS_avg_seq in {log_path}")
    return fps_avg


def load_entries(repo_root: Path) -> list[FpsEntry]:
    entries = [
        FpsEntry(
            tracker=tracker,
            fps_avg=_load_summary_fps(repo_root / rel_path),
        )
        for tracker, rel_path in TRACKER_SUMMARY_PATHS.items()
    ]

    mytracker_log = repo_root / "MyECOTracker" / "lasot" / "result" / "lasot936" / "tracking.log"
    entries.append(FpsEntry(tracker="MyTracker", fps_avg=_load_mytracker_fps(mytracker_log)))
    return sorted(entries, key=lambda item: item.fps_avg, reverse=True)


def plot_entries(entries: list[FpsEntry], out_png: Path, my_tracker_label: str = "MyTracker") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [entry.tracker for entry in entries]
    fps_values = [entry.fps_avg for entry in entries]
    colors = ["#d55e00" if entry.tracker == my_tracker_label else "#4c78a8" for entry in entries]

    fig_height = max(8, 0.42 * len(entries) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    y_positions = range(len(entries))
    bars = ax.barh(y_positions, fps_values, color=colors, edgecolor="#1f1f1f", linewidth=0.7)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(trackers)
    ax.invert_yaxis()
    ax.set_xlabel("FPS_avg_seq")
    ax.set_title("LaSOT headtail40 FPS Avg Comparison: MyTracker vs OtherTracker")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(fps_values) * 1.18)

    for idx, value in enumerate(fps_values):
        ax.text(
            value + max(fps_values) * 0.012,
            idx,
            _format_fps(value),
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.text(0.5, 0.01, "Metric: FPS_avg_seq. MyTracker is highlighted in orange.", ha="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_png = repo_root / "overall_result" / "lasot" / "fps_avg" / "fps_avg_lasot_headtail40_mytracker_vs_othertracker.png"
    entries = load_entries(repo_root)
    plot_entries(entries, out_png)
    print(f"Wrote chart: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
