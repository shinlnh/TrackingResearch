from __future__ import annotations

import csv
from pathlib import Path


SCORE_KEY_MAP = {
    "AUC": "success_auc",
    "Success50": "success50",
    "Precision": "precision20",
}


def load_csv_by_tracker(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["tracker"]: row for row in rows}


def normalize_local_metric(value: str) -> str:
    metric = float(value)
    if metric > 1.0:
        metric /= 100.0
    return f"{metric:.6f}"


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

    merged_rows: list[dict[str, str]] = []

    def pick_metric(
        score_row: dict[str, str],
        fps_row: dict[str, str],
        metric_key: str,
    ) -> str:
        fps_valid_raw = fps_row.get("valid_sequences", "")
        fps_value = fps_row.get(metric_key, "")
        if fps_valid_raw and fps_value:
            fps_valid = int(float(fps_valid_raw))
            if 0 < fps_valid < 40:
                return normalize_local_metric(fps_value)
        return score_row[SCORE_KEY_MAP[metric_key]]

    for tracker, row in scores.items():
        fps_row = fps.get(tracker, {})
        merged_rows.append(
            {
                "tracker": tracker,
                "success_auc": pick_metric(row, fps_row, "AUC"),
                "success50": pick_metric(row, fps_row, "Success50"),
                "precision20": pick_metric(row, fps_row, "Precision"),
                "rank_auc": row["rank_auc"],
                "fps_final_weighted_by_frames": fps_row.get("FPS_final_weighted_by_frames", ""),
                "fps_avg_seq_local": fps_row.get("FPS_avg_seq", ""),
                "fps_median_seq_local": fps_row.get("FPS_median_seq", ""),
                "fps_valid_sequences_local": fps_row.get("valid_sequences", ""),
                "fps_source_relpath": fps_row.get("source_relpath", ""),
            }
        )

    merged_rows.sort(key=lambda item: (-float(item["success_auc"]), item["tracker"]))
    for rank, row in enumerate(merged_rows, start=1):
        row["rank_auc"] = str(rank)

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Wrote merged metrics: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
