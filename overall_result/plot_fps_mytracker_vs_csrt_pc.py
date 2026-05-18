"""Render PC MyTracker vs Pure CSRT FPS challenge-representative plots.

Mirrors ``plot_mytracker_vs_csrt_pc.py`` but for FPS instead of AUC. Writes:

- overall_result/fps_challenge_representative_mytracker_vs_csrt.png
- overall_result/lasot_fps_challenge_representative_mytracker_vs_csrt.png

OTB100 + LaSOT both use the PC/local MyTracker timing exports (NOT Jetson
dual-acc). MyTracker times come from the verified_otb936_936 PC run.
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

# Common PC MyTracker timing source (works for both OTB and LaSOT seqs)
MYTRACKER_TIME_DIR = (
    REPO_ROOT
    / "MyECOTracker"
    / "pytracking"
    / "pytracking"
    / "tracking_results"
    / "eco"
    / "verified_otb936_936"
)


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    out_stem: str
    csrt_time_dir: Path
    my_dataset_fps: float
    csrt_dataset_fps: float
    sequences: tuple[dict[str, str], ...]


OTB_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT FPS on Representative OTB Challenge Sequences (PC)",
    out_stem="fps_challenge_representative_mytracker_vs_csrt",
    csrt_time_dir=REPO_ROOT
    / "OtherTracker"
    / "CSRT"
    / "otb100_results_full_20260326"
    / "txt_results"
    / "CSRT",
    my_dataset_fps=63.43,
    csrt_dataset_fps=92.98,
    sequences=(
        {"challenge_code": "OCC", "title": "Occlusion (OCC)", "sequence": "Bolt"},
        {"challenge_code": "IV", "title": "Illumination Variation (IV)", "sequence": "Human8"},
        {"challenge_code": "FM", "title": "Fast Motion (FM)", "sequence": "BlurCar3"},
    ),
)


LASOT_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT FPS on Representative LaSOT Challenge Sequences (PC)",
    out_stem="lasot_fps_challenge_representative_mytracker_vs_csrt",
    csrt_time_dir=REPO_ROOT
    / "OtherTracker"
    / "lasot"
    / "lasot936"
    / "CSRT"
    / "tracking_results"
    / "CSRT",
    my_dataset_fps=74.79,
    csrt_dataset_fps=112.18,
    sequences=(
        {"challenge_code": "POC/FOC", "title": "Occlusion (POC/FOC)", "sequence": "basketball-7"},
        {"challenge_code": "IV", "title": "Illumination Variation (IV)", "sequence": "shark-5"},
        {"challenge_code": "FM", "title": "Fast Motion (FM)", "sequence": "airplane-13"},
    ),
)


MY_COLOR = "#1f77b4"
CSRT_COLOR = "#ff7f0e"


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


def render_dataset(spec: DatasetSpec) -> None:
    out_png = OUT_DIR / f"{spec.out_stem}.png"
    out_csv = OUT_DIR / f"{spec.out_stem}.csv"
    csv_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 13.625),
        dpi=160,
        gridspec_kw={"width_ratios": [1.38, 1.0], "hspace": 0.34, "wspace": 0.18},
    )
    fig.suptitle(spec.title, fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.945, bottom=0.055, wspace=0.22, hspace=0.42)

    for row_idx, seq_spec in enumerate(spec.sequences):
        sequence = seq_spec["sequence"]
        my_path = MYTRACKER_TIME_DIR / f"{sequence}_time.txt"
        csrt_path = spec.csrt_time_dir / f"{sequence}_time.txt"

        my_times = load_times(my_path)
        csrt_times = load_times(csrt_path)
        my_fps = fps_per_frame(my_times)
        csrt_fps = fps_per_frame(csrt_times)

        # Skip first frame (init) when computing mean and plotting steady-state.
        my_fps_steady = my_fps[1:] if len(my_fps) > 1 else my_fps
        csrt_fps_steady = csrt_fps[1:] if len(csrt_fps) > 1 else csrt_fps
        my_mean = float(np.mean(my_fps_steady[my_fps_steady > 0])) if np.any(my_fps_steady > 0) else 0.0
        csrt_mean = float(np.mean(csrt_fps_steady[csrt_fps_steady > 0])) if np.any(csrt_fps_steady > 0) else 0.0
        delta = my_mean - csrt_mean

        ax_line = axes[row_idx, 0]
        frame_axis_my = np.arange(2, len(my_fps) + 1)
        frame_axis_csrt = np.arange(2, len(csrt_fps) + 1)
        ax_line.plot(frame_axis_my, my_fps[1:], color=MY_COLOR, linewidth=0.7, label="CA-CSRT")
        ax_line.plot(frame_axis_csrt, csrt_fps[1:], color=CSRT_COLOR, linewidth=0.7, label="Pure CSRT")
        ax_line.axhline(my_mean, color=MY_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
                        label=f"CA-CSRT mean ({my_mean:.1f})")
        ax_line.axhline(csrt_mean, color=CSRT_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
                        label=f"Pure CSRT mean ({csrt_mean:.1f})")
        ax_line.set_xlim(2, max(len(my_fps), len(csrt_fps)))
        ax_line.set_ylim(bottom=0)
        ax_line.set_xlabel("Frame", fontsize=8)
        ax_line.set_ylabel("Instantaneous FPS (frame init excluded)", fontsize=8)
        ax_line.grid(True, linewidth=0.35, alpha=0.35)
        ax_line.tick_params(labelsize=7)
        ax_line.set_title(
            f"{seq_spec['title']} - {sequence}\n"
            f"Mean FPS: CA-CSRT {my_mean:.1f} | CSRT {csrt_mean:.1f} | Delta {delta:+.1f}",
            fontsize=9,
            pad=6,
        )
        ax_line.legend(loc="best", fontsize=7, framealpha=0.9)

        ax_bar = axes[row_idx, 1]
        labels = ["CA-CSRT", "Pure CSRT"]
        values = [my_mean, csrt_mean]
        colors = [MY_COLOR, CSRT_COLOR]
        bars = ax_bar.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        max_val = max(values) if values else 1.0
        ax_bar.set_xlim(0, max_val * 1.25)
        ax_bar.set_xlabel("Mean FPS for this sequence (frames/sec)", fontsize=8)
        ax_bar.grid(True, axis="x", linewidth=0.35, alpha=0.35)
        ax_bar.tick_params(labelsize=8)
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
        dataset_label = "OTB100" if "OTB" in spec.title else "LaSOT"
        ax_bar.set_title(
            f"Per-sequence Mean FPS - {sequence}\n"
            f"(Dataset {dataset_label} avg: CA-CSRT {spec.my_dataset_fps:.2f}, "
            f"CSRT {spec.csrt_dataset_fps:.2f})",
            fontsize=9,
            pad=6,
        )

        csv_rows.append(
            {
                "challenge_code": seq_spec["challenge_code"],
                "sequence": sequence,
                "frames": min(len(my_times), len(csrt_times)),
                "mytracker_mean_fps_seq_pc": my_mean,
                "csrt_mean_fps_seq": csrt_mean,
                "delta_fps_my_minus_csrt": delta,
                "mytracker_dataset_fps_pc": spec.my_dataset_fps,
                "csrt_dataset_fps": spec.csrt_dataset_fps,
                "mytracker_time_file": rel(my_path),
                "csrt_time_file": rel(csrt_path),
            }
        )

    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {rel(out_png)}")
    print(f"wrote {rel(out_csv)}")
    for row in csv_rows:
        print(
            f"{row['sequence']}: "
            f"MyTracker={row['mytracker_mean_fps_seq_pc']:.1f}, "
            f"CSRT={row['csrt_mean_fps_seq']:.1f}, "
            f"Delta={row['delta_fps_my_minus_csrt']:+.1f}"
        )


def main() -> None:
    render_dataset(OTB_SPEC)
    render_dataset(LASOT_SPEC)


if __name__ == "__main__":
    main()
