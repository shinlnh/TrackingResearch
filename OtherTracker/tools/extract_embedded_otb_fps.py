from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_TRACKERS = [
    "CCOT",
    "DeepSRDCF",
    "SRDCFdecon",
    "SRDCF",
    "Staple",
    "HDT",
    "CF2",
    "LCT",
    "SAMF",
    "MEEM",
    "DSST",
    "KCF",
    "MDNet",
    "CNN-SVM",
]


def _unwrap_scalar(value) -> float:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return float(value)


def extract_summary(results_dir: Path, trackers: list[str]) -> list[dict[str, object]]:
    seq_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    for tracker in trackers:
        pattern = f"*_{tracker}.mat" if tracker != "CNN-SVM" else "*_CNN-SVM.mat"
        for mat_file in sorted(results_dir.glob(pattern)):
            data = loadmat(mat_file)
            result = data["results"][0, 0]
            fields = set(result.dtype.names or [])
            if "fps" not in fields or "len" not in fields:
                continue

            seq_name = mat_file.stem[: -(len(tracker) + 1)]
            fps = _unwrap_scalar(result["fps"])
            frames = int(round(_unwrap_scalar(result["len"])))
            if not math.isfinite(fps) or fps <= 0 or frames <= 0:
                continue

            seq_rows[tracker].append(
                {
                    "sequence": seq_name,
                    "fps": fps,
                    "frames": frames,
                    "time_sec": frames / fps,
                }
            )

    summary: list[dict[str, object]] = []
    for tracker in trackers:
        rows = seq_rows.get(tracker, [])
        if not rows:
            summary.append(
                {
                    "tracker": tracker,
                    "source": "embedded_otb_mat",
                    "status": "missing_embedded_fps",
                    "valid_sequences": 0,
                    "fps_avg_seq": math.nan,
                    "fps_median_seq": math.nan,
                    "fps_global": math.nan,
                    "fps_weighted_by_frames": math.nan,
                    "total_frames": 0,
                    "total_time_sec": math.nan,
                }
            )
            continue

        fps_values = np.array([r["fps"] for r in rows], dtype=np.float64)
        total_frames = int(sum(r["frames"] for r in rows))
        total_time = float(sum(r["time_sec"] for r in rows))
        fps_avg_seq = float(np.mean(fps_values))
        summary.append(
            {
                "tracker": tracker,
                "source": "embedded_otb_mat",
                "status": "ok",
                "valid_sequences": len(rows),
                "fps_avg_seq": fps_avg_seq,
                "fps_median_seq": float(np.median(fps_values)),
                "fps_global": fps_avg_seq,
                "fps_weighted_by_frames": float(total_frames / total_time),
                "total_frames": total_frames,
                "total_time_sec": total_time,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "source",
        "status",
        "valid_sequences",
        "fps_avg_seq",
        "fps_median_seq",
        "fps_global",
        "fps_weighted_by_frames",
        "total_frames",
        "total_time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--trackers", nargs="*", default=DEFAULT_TRACKERS)
    args = parser.parse_args()

    rows = extract_summary(args.results_dir, args.trackers)
    write_csv(args.out_csv, rows)
    print(f"Wrote {len(rows)} tracker summaries to {args.out_csv}")


if __name__ == "__main__":
    main()
