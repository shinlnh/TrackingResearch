from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from OtherTracker.tools.otb_sequences import OTBSequence, load_otb_sequences
from cnnsfm_approx import track_sequence


@dataclass(frozen=True)
class BenchmarkRow:
    sequence: str
    frames: int
    fps: float
    total_time_sec: float


def _slugify(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        cleaned.append(ch if ch.isalnum() else "_")
    return "".join(cleaned).strip("_")


def _filter_sequences(sequences: list[OTBSequence], selected_names: list[str] | None, limit: int | None) -> list[OTBSequence]:
    if selected_names:
        wanted = {name.lower() for name in selected_names}
        sequences = [seq for seq in sequences if seq.name.lower() in wanted]
    if limit is not None:
        sequences = sequences[:limit]
    if not sequences:
        raise ValueError("No sequences selected for benchmarking")
    return sequences


def _write_boxes(out_dir: Path, sequence: str, boxes: list[tuple[float, float, float, float]], times: list[float]) -> None:
    txt_dir = out_dir / "txt_results" / "CNN-SVM-Approx"
    txt_dir.mkdir(parents=True, exist_ok=True)

    with (txt_dir / f"{sequence}.txt").open("w", encoding="utf-8") as fh:
        for box in boxes:
            fh.write(",".join(f"{value:.6f}" for value in box) + "\n")

    with (txt_dir / f"{sequence}_time.txt").open("w", encoding="utf-8") as fh:
        for value in times:
            fh.write(f"{value:.9f}\n")


def _write_per_sequence_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frames", "fps", "total_time_sec"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "frames": row.frames,
                    "fps": row.fps,
                    "total_time_sec": row.total_time_sec,
                }
            )


def _write_summary_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    total_frames = sum(row.frames for row in rows)
    total_time = sum(row.total_time_sec for row in rows)
    fps_values = [row.fps for row in rows]
    summary = {
        "tracker": "CNN-SVM-Approx",
        "backend": "torchvision_sgdsvm",
        "parameter": "alexnet_fc6_saliency",
        "valid_sequences": len(rows),
        "fps_avg_seq": statistics.mean(fps_values),
        "fps_median_seq": statistics.median(fps_values),
        "fps_global": statistics.mean(fps_values),
        "fps_weighted_by_frames": total_frames / max(total_time, 1e-6),
        "total_frames": total_frames,
        "total_time_sec": total_time,
    }
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tracker",
                "backend",
                "parameter",
                "valid_sequences",
                "fps_avg_seq",
                "fps_median_seq",
                "fps_global",
                "fps_weighted_by_frames",
                "total_frames",
                "total_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--otb-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    sequences = load_otb_sequences(args.otb_root, args.sequence_file)
    sequences = _filter_sequences(sequences, args.sequences, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"[run ] {index}/{len(sequences)} {sequence.name} -> CNN-SVM-Approx")
        boxes, times = track_sequence(
            [str(path) for path in sequence.frame_paths()],
            sequence.init_rect,
            device=args.device,
            log_prefix=f"[CNN-SVM-Approx] {sequence.name}",
        )
        frames = len(sequence.frame_paths())
        total_time = sum(times[1:])
        fps = (frames - 1) / max(total_time, 1e-6)
        rows.append(BenchmarkRow(sequence=sequence.name, frames=frames, fps=fps, total_time_sec=total_time))
        _write_boxes(args.out_dir, sequence.name, boxes, times)
        print(f"[done] {sequence.name}: fps={fps:.6f}, time={total_time:.6f}s")

    base_name = _slugify("cnn_svm_approx")
    _write_per_sequence_csv(args.out_dir / f"{base_name}_per_sequence.csv", rows)
    _write_summary_csv(args.out_dir / f"{base_name}_summary.csv", rows)
    print(f"Wrote benchmark outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
