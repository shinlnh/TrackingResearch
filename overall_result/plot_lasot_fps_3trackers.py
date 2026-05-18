"""Plot LaSOT challenge-representative FPS comparisons for three trackers.

Mirrors ``plot_lasot_challenge_representative_3trackers.py`` but compares
FPS instead of AUC, between the PC MyTracker, OSTrack-384, and STARK-ST101
on the LaSOT head-tail-40 subset. Left panels show per-frame instantaneous
FPS, right panels show per-sequence mean FPS bar charts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "overall_result"
OUT_PNG = OUT_DIR / "lasot_fps_challenge_representative_mytracker_vs_ostrack_vs_stark.png"
OUT_CSV = OUT_DIR / "lasot_fps_challenge_representative_mytracker_vs_ostrack_vs_stark.csv"


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    label: str
    short_label: str
    color: str
    time_dir: Path
    dataset_fps_avg: float


TRACKERS = [
    TrackerSpec(
        key="myeco",
        label="CA-CSRT",
        short_label="CA-CSRT",
        color="#1f77b4",
        time_dir=REPO_ROOT
        / "MyECOTracker"
        / "pytracking"
        / "pytracking"
        / "tracking_results"
        / "eco"
        / "verified_otb936_936",
        dataset_fps_avg=74.79,
    ),
    TrackerSpec(
        key="ostrack",
        label="OSTrack-384",
        short_label="OSTrack",
        color="#d62728",
        time_dir=REPO_ROOT
        / "OtherTracker"
        / "lasot"
        / "lasot936"
        / "OSTrack"
        / "tracking_results"
        / "OSTrack",
        dataset_fps_avg=31.01,
    ),
    TrackerSpec(
        key="stark",
        label="STARK-ST101",
        short_label="STARK",
        color="#2ca02c",
        time_dir=REPO_ROOT
        / "OtherTracker"
        / "lasot"
        / "lasot936"
        / "STARK"
        / "tracking_results"
        / "STARK",
        dataset_fps_avg=43.76,
    ),
]


SEQUENCES = [
    {"challenge_code": "POC/FOC", "title": "Occlusion (POC/FOC)", "sequence": "basketball-7"},
    {"challenge_code": "IV", "title": "Illumination Variation (IV)", "sequence": "shark-5"},
    {"challenge_code": "FM", "title": "Fast Motion (FM)", "sequence": "airplane-13"},
]


def load_times(path: Path) -> np.ndarray:
    rows: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(float(line.split()[0]))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no timing rows found in {path}")
    return np.asarray(rows, dtype=float)


def fps_per_frame(times: np.ndarray) -> np.ndarray:
    fps = np.zeros_like(times)
    valid = times > 0
    fps[valid] = 1.0 / times[valid]
    return fps


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def plot() -> None:
    csv_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 13.625),
        dpi=160,
        gridspec_kw={"width_ratios": [1.38, 1.0], "hspace": 0.34, "wspace": 0.18},
    )
    fig.suptitle(
        "CA-CSRT vs OSTrack vs STARK FPS on Representative LaSOT Challenge Sequences (PC)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.945, bottom=0.055, wspace=0.22, hspace=0.42)

    for row_idx, spec in enumerate(SEQUENCES):
        sequence = spec["sequence"]

        per_tracker_times: dict[str, np.ndarray] = {}
        per_tracker_fps: dict[str, np.ndarray] = {}
        per_tracker_mean: dict[str, float] = {}
        per_tracker_files: dict[str, Path] = {}

        for tracker in TRACKERS:
            time_path = tracker.time_dir / f"{sequence}_time.txt"
            times = load_times(time_path)
            per_tracker_times[tracker.key] = times
            per_tracker_files[tracker.key] = time_path
            fps = fps_per_frame(times)
            per_tracker_fps[tracker.key] = fps
            steady = fps[1:] if len(fps) > 1 else fps
            per_tracker_mean[tracker.key] = (
                float(np.mean(steady[steady > 0])) if np.any(steady > 0) else 0.0
            )

        ax_line = axes[row_idx, 0]
        max_frames = 0
        for tracker in TRACKERS:
            fps = per_tracker_fps[tracker.key]
            mean_val = per_tracker_mean[tracker.key]
            frames = np.arange(2, len(fps) + 1)
            ax_line.plot(frames, fps[1:], color=tracker.color, linewidth=0.6, label=tracker.label)
            ax_line.axhline(
                mean_val,
                color=tracker.color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label=f"{tracker.short_label} mean ({mean_val:.1f})",
            )
            max_frames = max(max_frames, len(fps))
        ax_line.set_xlim(2, max_frames)
        ax_line.set_ylim(bottom=0)
        ax_line.set_xlabel("Frame", fontsize=8)
        ax_line.set_ylabel("Instantaneous FPS (frame init excluded)", fontsize=8)
        ax_line.grid(True, linewidth=0.35, alpha=0.35)
        ax_line.tick_params(labelsize=7)
        mean_text = " | ".join(
            f"{t.short_label} {per_tracker_mean[t.key]:.1f}" for t in TRACKERS
        )
        ax_line.set_title(
            f"{spec['title']} - {sequence}\nMean FPS: {mean_text}",
            fontsize=9,
            pad=6,
        )
        ax_line.legend(loc="best", fontsize=7, framealpha=0.9, ncol=2)

        ax_bar = axes[row_idx, 1]
        labels = [t.label for t in TRACKERS]
        values = [per_tracker_mean[t.key] for t in TRACKERS]
        colors = [t.color for t in TRACKERS]
        bars = ax_bar.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        max_val = max(values) if values else 1.0
        ax_bar.set_xlim(0, max_val * 1.25)
        ax_bar.set_xlabel("Mean FPS for this sequence (frames/sec)", fontsize=8)
        ax_bar.grid(True, axis="x", linewidth=0.35, alpha=0.35)
        ax_bar.tick_params(labelsize=8)
        ax_bar.invert_yaxis()
        for bar, val in zip(bars, values):
            ax_bar.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )
        dataset_avg = " / ".join(
            f"{t.short_label} {t.dataset_fps_avg:.2f}" for t in TRACKERS
        )
        ax_bar.set_title(
            f"Per-sequence Mean FPS - {sequence}\n(LaSOT dataset avg: {dataset_avg})",
            fontsize=9,
            pad=6,
        )

        csv_rows.append(
            {
                "challenge_code": spec["challenge_code"],
                "sequence": sequence,
                "frames": min(len(v) for v in per_tracker_times.values()),
                "mytracker_mean_fps_seq_pc": per_tracker_mean["myeco"],
                "ostrack_mean_fps_seq": per_tracker_mean["ostrack"],
                "stark_mean_fps_seq": per_tracker_mean["stark"],
                "mytracker_dataset_fps_pc": TRACKERS[0].dataset_fps_avg,
                "ostrack_dataset_fps": TRACKERS[1].dataset_fps_avg,
                "stark_dataset_fps": TRACKERS[2].dataset_fps_avg,
                "mytracker_time_file": rel(per_tracker_files["myeco"]),
                "ostrack_time_file": rel(per_tracker_files["ostrack"]),
                "stark_time_file": rel(per_tracker_files["stark"]),
            }
        )

    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {rel(OUT_PNG)}")
    print(f"wrote {rel(OUT_CSV)}")
    for row in csv_rows:
        print(
            f"{row['sequence']}: "
            f"MyTracker={row['mytracker_mean_fps_seq_pc']:.1f}, "
            f"OSTrack={row['ostrack_mean_fps_seq']:.1f}, "
            f"STARK={row['stark_mean_fps_seq']:.1f}"
        )


if __name__ == "__main__":
    plot()
