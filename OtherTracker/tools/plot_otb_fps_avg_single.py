from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FpsEntry:
    tracker: str
    fps_value: float
    source_type: str


def _format_fps(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def load_entries(csv_path: Path) -> list[FpsEntry]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    entries = [
        FpsEntry(
            tracker=row["tracker"].strip(),
            fps_value=float(row["fps_global"]),
            source_type=row.get("source_type", "").strip(),
        )
        for row in rows
    ]
    return sorted(entries, key=lambda item: item.fps_value, reverse=True)


def plot_entries(entries: list[FpsEntry], out_png: Path, my_tracker_label: str = "MyTracker") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [entry.tracker for entry in entries]
    fps_values = [entry.fps_value for entry in entries]
    colors = ["#d55e00" if entry.tracker == my_tracker_label else "#4c78a8" for entry in entries]

    hatches = []
    for entry in entries:
        if entry.source_type == "othertracker_embedded_fallback":
            hatches.append("//")
        elif entry.source_type == "othertracker_local_partial":
            hatches.append("xx")
        else:
            hatches.append("")

    fig_height = max(8, 0.42 * len(entries) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    y_positions = range(len(entries))
    bars = ax.barh(y_positions, fps_values, color=colors, edgecolor="#1f1f1f", linewidth=0.7)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(trackers)
    ax.invert_yaxis()
    ax.set_xlabel("FPS_avg")
    ax.set_title("OTB100 FPS Avg Comparison: MyTracker vs OtherTracker")
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

    fig.text(
        0.5,
        0.01,
        "MyTracker is highlighted in orange. Hatched bars indicate embedded fallback or partial-scope proxy benchmarks.",
        ha="center",
        fontsize=10,
    )

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "OtherTracker" / "fps_comparison_with_myecotracker.csv"
    out_png = repo_root / "overall_result" / "otb" / "fps_avg" / "fps_avg_mytracker_vs_othertracker.png"

    entries = load_entries(csv_path)
    plot_entries(entries, out_png)
    print(f"Wrote chart: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
