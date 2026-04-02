from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FpsEntry:
    tracker: str
    fps_avg: float
    note: str = ""
    source_type: str = ""


LASOT_SUMMARY_PATHS = {
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate single-panel FPS avg chart for video validation.")
    parser.add_argument("--dataset", choices=("otb", "lasot"), required=True)
    parser.add_argument("--scope", choices=("only_mytracker", "full_tracker"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args()


def _format_fps(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_otb_entries(repo_root: Path, scope: str) -> list[FpsEntry]:
    csv_path = repo_root / "OtherTracker" / "fps_comparison_with_myecotracker.csv"
    rows = _read_csv_rows(csv_path)
    entries = [
        FpsEntry(
            tracker=row["tracker"].strip(),
            fps_avg=float(row["fps_global"]),
            note=row.get("note", "").strip(),
            source_type=row.get("source_type", "").strip(),
        )
        for row in rows
    ]
    if scope == "only_mytracker":
        entries = [entry for entry in entries if entry.tracker == "MyTracker"]
    return sorted(entries, key=lambda item: item.fps_avg, reverse=True)


def _load_lasot_entries(repo_root: Path, scope: str) -> list[FpsEntry]:
    entries = []
    for tracker, rel_path in LASOT_SUMMARY_PATHS.items():
        row = _read_csv_rows(repo_root / rel_path)[0]
        entries.append(
            FpsEntry(
                tracker=tracker,
                fps_avg=float(row["FPS_avg_seq"]),
            )
        )

    my_log = repo_root / "MyECOTracker" / "lasot" / "result" / "lasot936" / "tracking.log"
    entries.append(FpsEntry(tracker="MyTracker", fps_avg=_load_mytracker_fps_from_log(my_log)))

    if scope == "only_mytracker":
        entries = [entry for entry in entries if entry.tracker == "MyTracker"]
    return sorted(entries, key=lambda item: item.fps_avg, reverse=True)


def _load_mytracker_fps_from_log(log_path: Path) -> float:
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = log_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Could not decode log file: {log_path}")

    fps_avg = None
    for line in text.splitlines():
        if line.startswith("FPS_avg_seq:"):
            fps_avg = float(line.split(":", 1)[1].strip())
    if fps_avg is None:
        raise ValueError(f"Could not find FPS_avg_seq in {log_path}")
    return fps_avg


def write_values_csv(entries: list[FpsEntry], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tracker", "fps_avg", "source_type", "note"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "tracker": entry.tracker,
                    "fps_avg": f"{entry.fps_avg:.6f}",
                    "source_type": entry.source_type,
                    "note": entry.note,
                }
            )


def plot_entries(entries: list[FpsEntry], out_png: Path, dataset: str, scope: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [entry.tracker for entry in entries]
    fps_values = [entry.fps_avg for entry in entries]
    colors = ["#d55e00" if entry.tracker == "MyTracker" else "#4c78a8" for entry in entries]

    hatches = []
    for entry in entries:
        if entry.source_type == "othertracker_embedded_fallback":
            hatches.append("//")
        elif entry.source_type == "othertracker_local_partial":
            hatches.append("xx")
        else:
            hatches.append("")

    fig_height = max(4.5, 0.42 * len(entries) + 2.0)
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
    ax.set_title(f"{dataset.upper()} FPS Avg: {scope}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(fps_values) * 1.18 if fps_values else 1)

    for idx, value in enumerate(fps_values):
        ax.text(
            value + max(fps_values) * 0.012,
            idx,
            _format_fps(value),
            va="center",
            ha="left",
            fontsize=10,
        )

    footer = "Metric: FPS_avg. MyTracker is highlighted in orange."
    if dataset == "otb" and scope == "full_tracker":
        footer += " Hatched bars indicate embedded fallback or partial-scope proxy benchmarks."
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    if args.dataset == "otb":
        entries = _load_otb_entries(args.repo_root, args.scope)
    else:
        entries = _load_lasot_entries(args.repo_root, args.scope)

    out_png = args.out_dir / "fps_avg.png"
    out_csv = args.out_dir / "fps_avg_values.csv"

    write_values_csv(entries, out_csv)
    plot_entries(entries, out_png, args.dataset, args.scope)
    print(f"Wrote chart: {out_png}")
    print(f"Wrote values: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
