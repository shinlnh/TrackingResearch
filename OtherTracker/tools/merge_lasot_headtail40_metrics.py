from __future__ import annotations

import csv
from pathlib import Path


def load_csv_by_tracker(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["tracker"]: row for row in rows}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    score_csv = repo_root / "ls" / "LaSOT_Evaluation_Toolkit" / "res_fig" / "headtail40_all_tracking_results_scores.csv"
    fps_csv = repo_root / "overall_result" / "video" / "lasot" / "full_tracker" / "fps_final" / "fps_final_values.csv"
    out_dir = repo_root / "overall_result" / "video" / "lasot" / "full_tracker" / "headtail40_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "headtail40_metrics_with_fps.csv"

    scores = load_csv_by_tracker(score_csv)
    fps = load_csv_by_tracker(fps_csv) if fps_csv.exists() else {}

    fieldnames = [
        "tracker",
        "success_auc",
        "success50",
        "precision20",
        "rank_auc",
        "fps_final_weighted_by_frames",
        "fps_avg_seq_local",
        "fps_median_seq_local",
        "fps_valid_sequences_local",
        "fps_source_relpath",
    ]

    def sort_key(item: tuple[str, dict[str, str]]) -> tuple[float, str]:
        tracker, row = item
        return (-float(row["success_auc"]), tracker)

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tracker, row in sorted(scores.items(), key=sort_key):
            fps_row = fps.get(tracker, {})
            writer.writerow(
                {
                    "tracker": tracker,
                    "success_auc": row["success_auc"],
                    "success50": row["success50"],
                    "precision20": row["precision20"],
                    "rank_auc": row["rank_auc"],
                    "fps_final_weighted_by_frames": fps_row.get("FPS_final_weighted_by_frames", ""),
                    "fps_avg_seq_local": fps_row.get("FPS_avg_seq", ""),
                    "fps_median_seq_local": fps_row.get("FPS_median_seq", ""),
                    "fps_valid_sequences_local": fps_row.get("valid_sequences", ""),
                    "fps_source_relpath": fps_row.get("source_relpath", ""),
                }
            )

    print(f"Wrote merged metrics: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
