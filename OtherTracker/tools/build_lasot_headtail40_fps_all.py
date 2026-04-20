from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_OVERRIDES = {
    "HCFT": "OtherTracker/lasot/lasot936/HCFT/summary_full40_check.csv",
}

SKIP_DIRS = {"benchmark17"}
SKIP_SUFFIXES = ("_smoke", "_debug", "_psverify")


def _pick_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return math.nan


def _pick_int(row: dict[str, str], *keys: str) -> int:
    value = _pick_float(row, *keys)
    if math.isnan(value):
        return 0
    return int(round(value))


def load_summary(path: Path, tracker: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    return {
        "tracker": tracker,
        "scope": row.get("scope", "lasot"),
        "valid_sequences": _pick_int(row, "valid_sequences", "ValidSequences"),
        "auc": _pick_float(row, "AUC", "auc"),
        "precision": _pick_float(row, "Precision", "precision"),
        "success50": _pick_float(row, "Success50", "success50"),
        "fps_avg_seq": _pick_float(row, "FPS_avg_seq", "fps_avg_seq"),
        "fps_median_seq": _pick_float(row, "FPS_median_seq", "fps_median_seq"),
        "fps_final": _pick_float(row, "FPS_weighted_by_frames", "fps_weighted_by_frames", "FPS_global", "fps_global"),
        "total_frames": _pick_int(row, "total_frames", "TotalFrames"),
        "total_time_sec": _pick_float(row, "total_time_sec", "TotalTimeSec"),
        "source_relpath": str(path),
    }


def load_mytracker_log(path: Path) -> dict[str, object]:
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Could not decode {path}")

    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    return {
        "tracker": "MyTracker",
        "scope": "lasot",
        "valid_sequences": int(values["valid_sequences"]),
        "auc": float(values["AUC"]),
        "precision": float(values["Precision"]),
        "success50": float(values["Success50"]),
        "fps_avg_seq": float(values["FPS_avg_seq"]),
        "fps_median_seq": float(values["FPS_median_seq"]),
        "fps_final": float(values["FPS_weighted_by_frames"]),
        "total_frames": int(values["total_frames"]),
        "total_time_sec": float(values["total_time_sec"]),
        "source_relpath": str(path),
    }


def discover_entries(repo_root: Path) -> list[dict[str, object]]:
    base_dir = repo_root / "OtherTracker" / "lasot" / "lasot936"
    entries: list[dict[str, object]] = []

    for tracker_dir in sorted(base_dir.iterdir()):
        if not tracker_dir.is_dir():
            continue
        if tracker_dir.name in SKIP_DIRS or tracker_dir.name.endswith(SKIP_SUFFIXES):
            continue

        override_relpath = SUMMARY_OVERRIDES.get(tracker_dir.name)
        summary_path = repo_root / override_relpath if override_relpath else tracker_dir / "summary.csv"
        if not summary_path.exists():
            continue

        entry = load_summary(summary_path, tracker_dir.name)
        entry["source_type"] = "summary_csv"
        entry["source_relpath"] = str(summary_path.relative_to(repo_root))
        entries.append(entry)

    mytracker_path = repo_root / "MyECOTracker" / "lasot" / "result" / "lasot936" / "tracking.log"
    entries.append(load_mytracker_log(mytracker_path))
    entries[-1]["source_type"] = "mytracker_log"
    entries[-1]["source_relpath"] = str(mytracker_path.relative_to(repo_root))

    return sorted(entries, key=lambda item: (float(item["fps_final"]), str(item["tracker"])), reverse=True)


def write_csv(entries: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "tracker",
        "valid_sequences",
        "AUC",
        "Precision",
        "Success50",
        "FPS_final_weighted_by_frames",
        "FPS_avg_seq",
        "FPS_median_seq",
        "total_frames",
        "total_time_sec",
        "source_type",
        "source_relpath",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, entry in enumerate(entries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "tracker": entry["tracker"],
                    "valid_sequences": entry["valid_sequences"],
                    "AUC": f"{float(entry['auc']):.6f}" if not math.isnan(float(entry["auc"])) else "",
                    "Precision": f"{float(entry['precision']):.6f}" if not math.isnan(float(entry["precision"])) else "",
                    "Success50": f"{float(entry['success50']):.6f}" if not math.isnan(float(entry["success50"])) else "",
                    "FPS_final_weighted_by_frames": f"{float(entry['fps_final']):.6f}",
                    "FPS_avg_seq": f"{float(entry['fps_avg_seq']):.6f}",
                    "FPS_median_seq": f"{float(entry['fps_median_seq']):.6f}",
                    "total_frames": entry["total_frames"],
                    "total_time_sec": f"{float(entry['total_time_sec']):.6f}",
                    "source_type": entry["source_type"],
                    "source_relpath": entry["source_relpath"],
                }
            )


def write_summary(entries: list[dict[str, object]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LaSOT all-tracker FPS report",
        "Metric: FPS_final_weighted_by_frames = total_frames / total_time_sec",
        f"Integrated trackers in this chart: {len(entries)}",
        "Notes: HCFT uses summary_full40_check.csv. PTAV uses its accepted 15-sequence representative subset.",
        "",
    ]
    for rank, entry in enumerate(entries, start=1):
        lines.append(
            f"{rank:02d}. {entry['tracker']}: valid_sequences={int(entry['valid_sequences'])}, "
            f"FPS_final={float(entry['fps_final']):.6f}, FPS_avg_seq={float(entry['fps_avg_seq']):.6f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(entries: list[dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    fps_values = []
    colors = []
    for entry in entries:
        tracker = str(entry["tracker"])
        labels.append(tracker)
        fps_values.append(float(entry["fps_final"]))
        if tracker == "MyTracker":
            colors.append("#d55e00")
        else:
            colors.append("#4c78a8")

    fig_height = max(12, 0.32 * len(entries) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height), constrained_layout=True)

    y_positions = range(len(entries))
    ax.barh(list(y_positions), fps_values, color=colors, edgecolor="#1f1f1f", linewidth=0.6)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("FPS")
    ax.set_title("LaSOT FPS Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    max_fps = max(fps_values)
    for idx, value in enumerate(fps_values):
        ax.text(value * 1.08, idx, f"{value:.2f}", va="center", ha="left", fontsize=8)

    ax.set_xlim(max(min(fps_values) * 0.7, 0.5), max_fps * 1.8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    entries = discover_entries(repo_root)

    out_dir = repo_root / "overall_result" / "picture" / "lasot" / "fps_all"
    write_csv(entries, out_dir / "fps_all_values.csv")
    write_summary(entries, out_dir / "fps_all_summary.txt")
    plot(entries, out_dir / "fps_all_log.png")

    print(f"Wrote CSV: {out_dir / 'fps_all_values.csv'}")
    print(f"Wrote summary: {out_dir / 'fps_all_summary.txt'}")
    print(f"Wrote plot: {out_dir / 'fps_all_log.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
