"""Build comparison report: MyECOTracker vs OSTrack vs STARK vs SwinTrack.

Reads existing summary.csv files from each tracker's result dir and outputs:
- comparison_lasot_headtail40.csv
- comparison_otb100.csv
- comparison_lasot_headtail40.png (bar plot)
- comparison_otb100.png (bar plot)

Designed to be re-runnable: if a tracker's CSV is missing, that row is skipped
with a warning instead of failing.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SummaryRow:
    tracker: str
    auc: float
    precision: float
    success50: float
    fps_avg_seq: float
    fps_weighted: float
    valid_sequences: int
    source: str


def _read_csv_row(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else None


def _pick(row: dict, *keys: str, default: float = math.nan) -> float:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                continue
    return default


def _pick_int(row: dict, *keys: str, default: int = 0) -> int:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                # Some files have "40/40" format
                if "/" in str(v):
                    return int(str(v).split("/")[0])
                return int(round(float(v)))
            except (ValueError, AttributeError):
                continue
    return default


def load_summary(path: Path, label: str) -> Optional[SummaryRow]:
    row = _read_csv_row(path)
    if row is None:
        return None
    return SummaryRow(
        tracker=label,
        auc=_pick(row, "AUC", "AUC_mean", "auc"),
        precision=_pick(row, "Precision", "Precision_mean", "precision"),
        success50=_pick(row, "Success50", "Success50_mean", "success50"),
        fps_avg_seq=_pick(row, "FPS_avg_seq", "fps_avg_seq"),
        fps_weighted=_pick(row, "FPS_weighted_by_frames", "fps_weighted_by_frames"),
        valid_sequences=_pick_int(row, "valid_sequences", "ValidSequences"),
        source=str(path.relative_to(REPO_ROOT)),
    )


LASOT_SOURCES = [
    ("MyECOTracker",
     REPO_ROOT / "jetson_reports" / "verified_otb936_dual_acc_lasot_headtail40" / "summary.csv"),
    ("ToMP-50",
     REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "ToMP" / "summary.csv"),
    ("OSTrack-384",
     REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "OSTrack" / "summary.csv"),
    ("STARK-ST101",
     REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "STARK" / "summary.csv"),
    ("SwinTrack",
     REPO_ROOT / "OtherTracker" / "lasot" / "lasot936" / "SwinTrack" / "summary.csv"),
]

OTB_SOURCES = [
    ("MyECOTracker",
     REPO_ROOT / "overall_result" / "jetson_verified_otb936_dual_acc_otb100_full_20260404_1618" / "summary.csv"),
    ("OSTrack-384",
     REPO_ROOT / "OtherTracker" / "OSTrack" / "otb100_results" / "summary.csv"),
    ("STARK-ST101",
     REPO_ROOT / "OtherTracker" / "Stark" / "otb100_results" / "summary.csv"),
    ("SwinTrack",
     REPO_ROOT / "OtherTracker" / "SwinTrack" / "otb100_results" / "summary.csv"),
]


def write_comparison_csv(path: Path, rows: list[SummaryRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["Tracker", "AUC", "Precision", "Success50", "FPS_avg_seq",
             "FPS_weighted_by_frames", "valid_sequences", "source"]
        )
        for r in rows:
            writer.writerow(
                [r.tracker, f"{r.auc:.2f}", f"{r.precision:.2f}",
                 f"{r.success50:.2f}", f"{r.fps_avg_seq:.2f}",
                 f"{r.fps_weighted:.2f}", r.valid_sequences, r.source]
            )


def plot_comparison(rows: list[SummaryRow], title: str, out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    if not rows:
        print(f"no data, skip plot for {title}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r.tracker for r in rows]
    auc_vals = [r.auc for r in rows]
    fps_vals = [r.fps_avg_seq for r in rows]

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"][: len(rows)]

    ax = axes[0]
    bars = ax.bar(labels, auc_vals, color=colors)
    ax.set_ylabel("AUC (%)")
    ax.set_title(f"{title} — Success AUC")
    ax.set_ylim(0, max(100, max(auc_vals) * 1.1))
    for b, v in zip(bars, auc_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    bars = ax.bar(labels, fps_vals, color=colors)
    ax.set_ylabel("FPS (avg per sequence)")
    ax.set_title(f"{title} — FPS")
    ax.set_ylim(0, max(fps_vals) * 1.2 if fps_vals else 1)
    for b, v in zip(bars, fps_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote plot: {out_path.relative_to(REPO_ROOT)}")


def main(argv: list[str]) -> int:
    out_dir = REPO_ROOT / "overall_result" / "sota_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== LaSOT head-tail-40 ===")
    lasot_rows: list[SummaryRow] = []
    for label, path in LASOT_SOURCES:
        row = load_summary(path, label)
        if row is None:
            print(f"  [missing] {label}: {path.relative_to(REPO_ROOT)}")
            continue
        lasot_rows.append(row)
        print(f"  [ok]      {label}: AUC={row.auc:.2f}  P20={row.precision:.2f}  "
              f"FPS={row.fps_avg_seq:.2f}  n={row.valid_sequences}")

    write_comparison_csv(out_dir / "comparison_lasot_headtail40.csv", lasot_rows)
    plot_comparison(lasot_rows, "LaSOT head-tail-40",
                    out_dir / "comparison_lasot_headtail40.png")

    print()
    print("=== OTB-100 ===")
    otb_rows: list[SummaryRow] = []
    for label, path in OTB_SOURCES:
        row = load_summary(path, label)
        if row is None:
            print(f"  [missing] {label}: {path.relative_to(REPO_ROOT)}")
            continue
        otb_rows.append(row)
        print(f"  [ok]      {label}: AUC={row.auc:.2f}  P20={row.precision:.2f}  "
              f"FPS={row.fps_avg_seq:.2f}  n={row.valid_sequences}")

    write_comparison_csv(out_dir / "comparison_otb100.csv", otb_rows)
    plot_comparison(otb_rows, "OTB-100", out_dir / "comparison_otb100.png")

    print()
    print(f"output dir: {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
