from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


PRECISION20_INDEX = 20


@dataclass(frozen=True)
class PrecisionEntry:
    tracker: str
    precision_percent: float


def _format_precision(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def load_lasot_entries(repo_root: Path) -> list[PrecisionEntry]:
    metrics_path = (
        repo_root
        / "overall_result"
        / "video"
        / "lasot"
        / "full_tracker"
        / "headtail40_metrics"
        / "headtail40_metrics_with_fps.csv"
    )
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    entries = [
        PrecisionEntry(
            tracker=row["tracker"].strip(),
            precision_percent=float(row["precision20"]) * 100.0,
        )
        for row in rows
    ]
    return sorted(entries, key=lambda item: (item.precision_percent, item.tracker), reverse=True)


def load_otb_entries(repo_root: Path) -> list[PrecisionEntry]:
    perfmat_path = repo_root / "otb" / "otb-toolkit" / "perfmat" / "OPE" / "perfplot_curves_OPE.mat"
    data = sio.loadmat(perfmat_path, squeeze_me=True, struct_as_record=False)

    tracker_names = [str(name) for name in np.atleast_1d(data["nameTrkAll"]).tolist()]
    precision_curve = np.asarray(data["precision_curve"], dtype=object)

    entries: list[PrecisionEntry] = []
    for tracker_idx, tracker_name in enumerate(tracker_names):
        seq_curves = np.ravel(precision_curve[tracker_idx])
        precision20_values = []
        for seq_curve in seq_curves:
            curve = np.asarray(seq_curve, dtype=np.float64).reshape(-1)
            if curve.size <= PRECISION20_INDEX:
                continue
            precision20_values.append(float(curve[PRECISION20_INDEX]))

        if not precision20_values:
            continue

        entries.append(
            PrecisionEntry(
                tracker=tracker_name,
                precision_percent=float(np.mean(np.asarray(precision20_values, dtype=np.float64)) * 100.0),
            )
        )

    return sorted(entries, key=lambda item: (item.precision_percent, item.tracker), reverse=True)


def write_csv(entries: list[PrecisionEntry], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "tracker", "precision"])
        writer.writeheader()
        for rank, entry in enumerate(entries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "tracker": entry.tracker,
                    "precision": f"{entry.precision_percent:.6f}",
                }
            )


def write_summary(entries: list[PrecisionEntry], out_txt: Path, title: str) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, f"Integrated trackers in this chart: {len(entries)}", ""]
    for rank, entry in enumerate(entries, start=1):
        lines.append(f"{rank:02d}. {entry.tracker}: Precision={entry.precision_percent:.6f}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_entries(
    entries: list[PrecisionEntry],
    out_png: Path,
    *,
    title: str,
    mytracker_color: str,
    other_color: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [entry.tracker for entry in entries]
    values = [entry.precision_percent for entry in entries]
    colors = [mytracker_color if entry.tracker == "MyTracker" else other_color for entry in entries]

    fig_height = max(8, 0.42 * len(entries) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    y_positions = range(len(entries))
    ax.barh(y_positions, values, color=colors, edgecolor="#1f1f1f", linewidth=0.7)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(trackers)
    ax.invert_yaxis()
    ax.set_xlabel("Precision")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.18)

    offset = max(values) * 0.012
    for idx, value in enumerate(values):
        ax.text(offset + value, idx, _format_precision(value), va="center", ha="left", fontsize=10)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_otb_outputs(repo_root: Path) -> None:
    entries = load_otb_entries(repo_root)
    out_dir = repo_root / "overall_result" / "picture" / "otb" / "precision"
    write_csv(entries, out_dir / "precision_values.csv")
    write_summary(entries, out_dir / "precision_summary.txt", "OTB100 Precision Comparison")
    plot_entries(
        entries,
        out_dir / "precision_mytracker_vs_othertracker.png",
        title="OTB100 Precision Comparison",
        mytracker_color="#f2c94c",
        other_color="#ef7aa1",
    )


def build_lasot_outputs(repo_root: Path) -> None:
    entries = load_lasot_entries(repo_root)
    out_dir = repo_root / "overall_result" / "picture" / "lasot" / "precision"
    write_csv(entries, out_dir / "precision_values.csv")
    write_summary(entries, out_dir / "precision_summary.txt", "LaSOT Precision Comparison")
    plot_entries(
        entries,
        out_dir / "precision_all.png",
        title="LaSOT Precision Comparison",
        mytracker_color="#d55e00",
        other_color="#4c78a8",
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    build_otb_outputs(repo_root)
    build_lasot_outputs(repo_root)
    print("Wrote OTB100 precision chart + CSV + summary")
    print("Wrote LaSOT precision chart + CSV + summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
