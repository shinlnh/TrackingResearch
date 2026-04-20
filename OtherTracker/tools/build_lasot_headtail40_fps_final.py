from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrackerSource:
    tracker: str
    source_type: str
    rel_path: str


TRACKER_SOURCES = (
    TrackerSource("KCF", "summary_csv", "OtherTracker/lasot/lasot936/KCF/summary.csv"),
    TrackerSource("CSRT", "summary_csv", "OtherTracker/lasot/lasot936/CSRT/summary.csv"),
    TrackerSource("DSST", "summary_csv", "OtherTracker/lasot/lasot936/DSST/summary.csv"),
    TrackerSource("DeepSRDCF", "summary_csv", "OtherTracker/lasot/lasot936/DeepSRDCF/summary.csv"),
    TrackerSource("Staple", "summary_csv", "OtherTracker/lasot/lasot936/Staple/summary.csv"),
    TrackerSource("ECO", "summary_csv", "OtherTracker/lasot/lasot936/ECO/summary.csv"),
    TrackerSource("ToMP", "summary_csv", "OtherTracker/lasot/lasot936/ToMP/summary.csv"),
    TrackerSource("LCT", "summary_csv", "OtherTracker/lasot/lasot936/LCT/summary.csv"),
    TrackerSource("SAMF", "summary_csv", "OtherTracker/lasot/lasot936/SAMF/summary.csv"),
    TrackerSource("MEEM", "summary_csv", "OtherTracker/lasot/lasot936/MEEM/summary.csv"),
    TrackerSource("PTAV", "summary_csv", "OtherTracker/lasot/lasot936/PTAV/summary.csv"),
    TrackerSource("SRDCF", "summary_csv", "OtherTracker/lasot/lasot936/SRDCF/summary.csv"),
    TrackerSource("CNN-SVM", "summary_csv", "OtherTracker/lasot/lasot936/CNN-SVM/summary.csv"),
    TrackerSource("MDNet", "summary_csv", "OtherTracker/lasot/lasot936/MDNet/summary.csv"),
    TrackerSource("CF2", "summary_csv", "OtherTracker/lasot/lasot936/CF2/summary.csv"),
    TrackerSource("SRDCFdecon", "summary_csv", "OtherTracker/lasot/lasot936/SRDCFdecon/summary.csv"),
    TrackerSource("HDT", "summary_csv", "OtherTracker/lasot/lasot936/HDT/summary.csv"),
    TrackerSource("MyTracker", "mytracker_log", "MyECOTracker/lasot/result/lasot936/tracking.log"),
)

# These verified trees currently have a dedicated LaSOT headtail40 runner in this repo.
SUPPORTED_VERIFIED_DIRS = {
    "DSST": "DSST",
    "ECO-master": "ECO",
    "KCF": "KCF",
    "LCT": "LCT",
    "MDNet-master": "MDNet",
    "MEEM": "MEEM",
    "pytracking-master": "ToMP",
    "SAMF": "SAMF",
    "SRDCF": "SRDCF",
    "Staple": "Staple",
}

# These are runnable on LaSOT headtail40 today, but their source tree is outside OtherTracker/verified.
SUPPORTED_NON_VERIFIED = {
    "CF2",
    "CNN-SVM",
    "CSRT",
    "DeepSRDCF",
    "HDT",
    "MyTracker",
    "SRDCFdecon",
}


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


def _pick_text(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _load_summary_csv(path: Path, tracker: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    return {
        "tracker": tracker,
        "valid_sequences": _pick_int(row, "valid_sequences", "ValidSequences"),
        "auc": _pick_float(row, "AUC", "auc"),
        "precision": _pick_float(row, "Precision", "precision"),
        "success50": _pick_float(row, "Success50", "success50"),
        "fps_avg_seq": _pick_float(row, "FPS_avg_seq", "fps_avg_seq"),
        "fps_median_seq": _pick_float(row, "FPS_median_seq", "fps_median_seq"),
        "fps_final": _pick_float(row, "FPS_weighted_by_frames", "fps_weighted_by_frames", "FPS_global", "fps_global"),
        "total_frames": _pick_int(row, "total_frames", "TotalFrames"),
        "total_time_sec": _pick_float(row, "total_time_sec", "TotalTimeSec"),
        "source_relpath": path,
    }


def _load_mytracker_log(path: Path) -> dict[str, object]:
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
        key = key.strip()
        if key in {
            "valid_sequences",
            "AUC",
            "Precision",
            "Success50",
            "FPS_avg_seq",
            "FPS_median_seq",
            "FPS_weighted_by_frames",
            "total_frames",
            "total_time_sec",
        }:
            values[key] = value.strip()

    return {
        "tracker": "MyTracker",
        "valid_sequences": int(values["valid_sequences"]),
        "auc": float(values["AUC"]),
        "precision": float(values["Precision"]),
        "success50": float(values["Success50"]),
        "fps_avg_seq": float(values["FPS_avg_seq"]),
        "fps_median_seq": float(values["FPS_median_seq"]),
        "fps_final": float(values["FPS_weighted_by_frames"]),
        "total_frames": int(values["total_frames"]),
        "total_time_sec": float(values["total_time_sec"]),
        "source_relpath": path,
    }


def load_entries(repo_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for source in TRACKER_SOURCES:
        path = repo_root / source.rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        if source.source_type == "summary_csv":
            entry = _load_summary_csv(path, tracker=source.tracker)
        elif source.source_type == "mytracker_log":
            entry = _load_mytracker_log(path)
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
        entry["source_type"] = source.source_type
        entry["source_relpath"] = str(Path(source.rel_path))
        entries.append(entry)
    return sorted(entries, key=lambda item: float(item["fps_final"]), reverse=True)


def write_fps_csv(entries: list[dict[str, object]], out_csv: Path) -> None:
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
                    "AUC": f"{float(entry['auc']):.6f}",
                    "Precision": f"{float(entry['precision']):.6f}",
                    "Success50": f"{float(entry['success50']):.6f}",
                    "FPS_final_weighted_by_frames": f"{float(entry['fps_final']):.6f}",
                    "FPS_avg_seq": f"{float(entry['fps_avg_seq']):.6f}",
                    "FPS_median_seq": f"{float(entry['fps_median_seq']):.6f}",
                    "total_frames": entry["total_frames"],
                    "total_time_sec": f"{float(entry['total_time_sec']):.6f}",
                    "source_type": entry["source_type"],
                    "source_relpath": entry["source_relpath"],
                }
            )


def write_coverage_csv(repo_root: Path, out_csv: Path) -> None:
    verified_root = repo_root / "OtherTracker" / "verified"
    rows: list[dict[str, str]] = []
    for directory in sorted(p for p in verified_root.iterdir() if p.is_dir()):
        tracker = SUPPORTED_VERIFIED_DIRS.get(directory.name, "")
        if tracker:
            status = "integrated_headtail40_runner_present"
            note = f"Mapped to tracker label {tracker}"
        else:
            status = "source_present_but_no_headtail40_runner"
            note = "Needs LaSOT dataset adapter / runner integration before batch execution"
        rows.append(
            {
                "verified_dir": directory.name,
                "tracker_label": tracker,
                "status": status,
                "note": note,
            }
        )

    for tracker in sorted(SUPPORTED_NON_VERIFIED):
        rows.append(
            {
                "verified_dir": "",
                "tracker_label": tracker,
                "status": "headtail40_runner_present_outside_verified",
                "note": "Integrated in repo, but source tree is not under OtherTracker/verified",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["verified_dir", "tracker_label", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


def plot_fps(entries: list[dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [str(entry["tracker"]) for entry in entries]
    fps_values = [float(entry["fps_final"]) for entry in entries]
    colors = ["#d55e00" if tracker == "MyTracker" else "#4c78a8" for tracker in trackers]

    fig_height = max(8, 0.42 * len(entries) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    y_positions = range(len(entries))
    ax.barh(y_positions, fps_values, color=colors, edgecolor="#1f1f1f", linewidth=0.7)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(trackers)
    ax.invert_yaxis()
    ax.set_xlabel("FPS_final_weighted_by_frames")
    ax.set_title("LaSOT HeadTail40 Final FPS Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(fps_values) * 1.15)

    offset = max(fps_values) * 0.01
    for idx, value in enumerate(fps_values):
        ax.text(value + offset, idx, f"{value:.2f}", va="center", ha="left", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Metric = total_frames / total_time_sec over each tracker's available evaluated sequences. "
        "MyTracker is highlighted in orange; PTAV currently reflects its 15-sequence representative subset.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(entries: list[dict[str, object]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LaSOT headtail40 final FPS report",
        "Metric: FPS_final_weighted_by_frames = total_frames / total_time_sec",
        f"Integrated trackers with runnable headtail40 outputs: {len(entries)}",
        "Note: PTAV currently uses a 15-sequence representative subset rather than all 40 sequences.",
        "",
    ]
    for rank, entry in enumerate(entries, start=1):
        lines.append(
            f"{rank:02d}. {entry['tracker']}: valid_sequences={int(entry['valid_sequences'])}, "
            f"FPS_final={float(entry['fps_final']):.6f}, FPS_avg_seq={float(entry['fps_avg_seq']):.6f}, "
            f"AUC={float(entry['auc']):.6f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    entries = load_entries(repo_root)

    out_dir = repo_root / "overall_result" / "video" / "lasot" / "full_tracker" / "fps_final"
    write_fps_csv(entries, out_dir / "fps_final_values.csv")
    write_coverage_csv(repo_root, out_dir / "verified_coverage.csv")
    write_text_summary(entries, out_dir / "fps_final_summary.txt")
    plot_fps(entries, out_dir / "fps_final.png")

    print(f"Wrote CSV: {out_dir / 'fps_final_values.csv'}")
    print(f"Wrote coverage: {out_dir / 'verified_coverage.csv'}")
    print(f"Wrote summary: {out_dir / 'fps_final_summary.txt'}")
    print(f"Wrote plot: {out_dir / 'fps_final.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
