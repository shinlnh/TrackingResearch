from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from pathlib import Path
import sys

import numpy as np


def _bootstrap_pytracking(repo_root: Path):
    pytracking_root = repo_root / "OtherTracker" / "pytracking"
    root_str = str(pytracking_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from pytracking.evaluation import get_dataset
    return get_dataset


def _load_times(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", ndmin=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess OtherTracker ECO LaSOT results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--toolkit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--display-name", default="ECO")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    get_dataset = _bootstrap_pytracking(repo_root)
    dataset = list(get_dataset("lasot"))

    toolkit_tracker_dir = args.toolkit_dir / "tracking_results" / "ECO_tracking_result"
    toolkit_tracker_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    fps_values: list[float] = []
    total_frames = 0
    total_time = 0.0
    missing = []

    for seq in dataset:
        bbox_src = args.results_dir / f"{seq.name}.txt"
        time_src = args.results_dir / f"{seq.name}_time.txt"

        if not bbox_src.exists() or not time_src.exists():
            missing.append(seq.name)
            continue

        bbox_dst = toolkit_tracker_dir / bbox_src.name
        shutil.copy2(bbox_src, bbox_dst)

        times = _load_times(time_src)
        frames = int(times.size)
        seq_time = float(times.sum())
        fps = float(frames / seq_time) if seq_time > 0 else float("nan")

        total_frames += frames
        total_time += seq_time
        fps_values.append(fps)
        manifest_rows.append(
            {
                "sequence": seq.name,
                "frames": frames,
                "total_time_sec": f"{seq_time:.6f}",
                "fps": f"{fps:.6f}",
                "bbox_src": str(bbox_src),
                "time_src": str(time_src),
            }
        )

    manifest_path = args.out_dir / "tracking_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sequence", "frames", "total_time_sec", "fps", "bbox_src", "time_src"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "tracker": args.display_name,
        "dataset": "LaSOT test_set",
        "valid_sequences": len(manifest_rows),
        "missing_sequences": len(missing),
        "fps_avg_seq": statistics.mean(fps_values) if fps_values else float("nan"),
        "fps_median_seq": statistics.median(fps_values) if fps_values else float("nan"),
        "fps_weighted_by_frames": (total_frames / total_time) if total_time > 0 else float("nan"),
        "total_frames": total_frames,
        "total_time_sec": total_time,
        "results_dir": str(args.results_dir),
    }

    summary_path = args.out_dir / "fps_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    if missing:
        missing_path = args.out_dir / "missing_sequences.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing {len(missing)} sequences. See {missing_path}")
        return 1

    print(f"Synced {len(manifest_rows)} sequences into {toolkit_tracker_dir}")
    print(f"fps_avg_seq={summary['fps_avg_seq']:.6f}")
    print(f"fps_weighted_by_frames={summary['fps_weighted_by_frames']:.6f}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
