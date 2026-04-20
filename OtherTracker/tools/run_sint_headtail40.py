import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
SINT_DIR = THIS_DIR.parent / "verified" / "SINT-master"
sys.path.insert(0, str(SINT_DIR))

from run_sint_lasot import get_runtime_desc, run_sequence


def read_sequence_names(sequence_file):
    return [
        line.strip()
        for line in Path(sequence_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-file", required=True)
    parser.add_argument("--lasot-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_names = read_sequence_names(args.sequence_file)
    start_index = max(1, args.start_index)
    end_index = min(len(sequence_names), args.end_index)
    lasot_root = Path(args.lasot_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sint] runtime={get_runtime_desc()}", flush=True)

    for idx in range(start_index, end_index + 1):
        seq_name = sequence_names[idx - 1]
        class_name = seq_name.split("-", 1)[0]
        seq_dir = lasot_root / class_name / seq_name
        bbox_file = results_dir / f"{seq_name}.txt"
        time_file = results_dir / f"{seq_name}_time.txt"
        if bbox_file.exists() and time_file.exists():
            print(f"[skip] {idx:3d}/{len(sequence_names):3d} {seq_name}", flush=True)
            continue

        print(f"[run ] {idx:3d}/{len(sequence_names):3d} {seq_name}", flush=True)
        _, times, fps = run_sequence(
            seq_dir,
            results_path=bbox_file,
            time_path=time_file,
            log_every=args.log_every,
            save_every=args.save_every,
        )
        print(
            f"[done] {idx:3d}/{len(sequence_names):3d} {seq_name} "
            f"fps={fps:.6f} total_time={times.sum():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
