from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OtherTracker.tools.extract_embedded_otb_fps import extract_summary
from OtherTracker.tools.otb_sequences import load_otb_sequences


def _matlab_quote(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _build_batch_expression(
    repo_root: Path,
    mode: str,
    otb_root: Path,
    sequence_file: Path,
    out_dir: Path,
    start_index: int,
    end_index: int | None,
) -> str:
    addpath_target = _matlab_quote(repo_root / "OtherTracker" / "CCOT")
    runner_name = "run_ccot_otb_gpu" if mode == "gpu" else "run_ccot_otb"
    end_token = "Inf" if end_index is None else str(end_index)

    parts = []
    if mode == "gpu":
        parts.append("parallel.gpu.enableCUDAForwardCompatibility(true)")
    parts.append(f"addpath('{addpath_target}')")
    parts.append(
        f"{runner_name}('{_matlab_quote(otb_root)}','{_matlab_quote(sequence_file)}',"
        f"'{_matlab_quote(out_dir)}',{start_index},{end_token})"
    )
    return "; ".join(parts)


def _ccot_mat_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("*_CCOT.mat"))


def _completed_sequence_names(out_dir: Path) -> set[str]:
    return {path.stem[: -len("_CCOT")] for path in _ccot_mat_files(out_dir)}


def _expected_sequence_count(sequence_file: Path | None) -> int | None:
    if sequence_file is None or not sequence_file.exists():
        return None
    return sum(1 for line in sequence_file.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize_results(out_dir: Path, source_label: str, sequence_file: Path | None = None) -> Path:
    rows = extract_summary(out_dir, ["CCOT"])
    if not rows:
        raise RuntimeError(f"No CCOT summary rows could be built from {out_dir}")

    row = rows[0]
    row["source"] = source_label
    expected_count = _expected_sequence_count(sequence_file)
    if expected_count is not None:
        valid_sequences = int(row["valid_sequences"])
        if valid_sequences == 0:
            row["status"] = "missing"
        elif valid_sequences < expected_count:
            row["status"] = "partial"
        else:
            row["status"] = "ok"

    out_path = out_dir / "ccot_summary.csv"
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
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return out_path


def _launch_background(command: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab") as stdout_fh, stderr_path.open("ab") as stderr_fh:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout_fh,
            stderr=stderr_fh,
            creationflags=(
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            ),
            close_fds=True,
        )
    return process.pid


def _run_foreground(command: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(command, cwd=str(cwd))
    return completed.returncode


def _run_sequence_at_a_time(
    *,
    matlab_bin: str,
    mode: str,
    otb_root: Path,
    sequence_file: Path,
    out_dir: Path,
    start_index: int,
    end_index: int | None,
) -> int:
    sequences = load_otb_sequences(otb_root, sequence_file)
    first_index = max(1, start_index)
    last_index = len(sequences) if end_index is None else min(len(sequences), end_index)

    completed_names = _completed_sequence_names(out_dir)
    for seq_index in range(first_index, last_index + 1):
        seq = sequences[seq_index - 1]
        result_path = out_dir / f"{seq.name}_CCOT.mat"
        if seq.name in completed_names and result_path.exists():
            print(f"[skip] {seq_index}/{len(sequences)} {seq.name}")
            continue

        batch_expr = _build_batch_expression(
            repo_root=REPO_ROOT,
            mode=mode,
            otb_root=otb_root.resolve(),
            sequence_file=sequence_file.resolve(),
            out_dir=out_dir.resolve(),
            start_index=seq_index,
            end_index=seq_index,
        )
        command = [matlab_bin, "-batch", batch_expr]
        print(f"[run ] {seq_index}/{len(sequences)} {seq.name} -> CCOT")
        exit_code = _run_foreground(command, REPO_ROOT)
        if exit_code != 0:
            print(f"[fail] {seq.name}: MATLAB exit code {exit_code}", file=sys.stderr)
            return exit_code

        summarize_results(out_dir, f"local_ccot_matlab_{mode}", sequence_file)
        print(f"[done] {seq.name}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Python driver that launches the official CCOT OTB benchmark and summarizes FPS."
    )
    parser.add_argument("--mode", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--otb-root", type=Path, default=REPO_ROOT / "otb" / "otb100")
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=REPO_ROOT / "otb" / "otb-toolkit" / "sequences" / "SEQUENCES",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "OtherTracker" / "CCOT" / "otb100_fps_gpu",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--sequence-at-a-time", action="store_true")
    args = parser.parse_args(argv)

    source_label = f"local_ccot_matlab_{args.mode}"

    if args.summarize_only:
        summary_path = summarize_results(args.out_dir, source_label, args.sequence_file)
        print(f"Existing CCOT result files: {len(_ccot_mat_files(args.out_dir))}")
        print(f"Wrote summary to {summary_path}")
        return 0

    batch_expr = _build_batch_expression(
        repo_root=REPO_ROOT,
        mode=args.mode,
        otb_root=args.otb_root.resolve(),
        sequence_file=args.sequence_file.resolve(),
        out_dir=args.out_dir.resolve(),
        start_index=args.start_index,
        end_index=args.end_index,
    )
    command = [args.matlab_bin, "-batch", batch_expr]

    if args.background:
        if args.sequence_at_a_time:
            raise SystemExit("--background and --sequence-at-a-time cannot be used together")
        log_base = args.out_dir / f"ccot_{args.mode}_python_driver"
        pid = _launch_background(
            command=command,
            cwd=REPO_ROOT,
            stdout_path=log_base.with_suffix(".log"),
            stderr_path=log_base.with_suffix(".err.log"),
        )
        pid_path = args.out_dir / f"ccot_{args.mode}_python_driver.pid"
        pid_path.write_text(str(pid), encoding="utf-8")
        print(f"Started CCOT {args.mode} run in background (PID {pid})")
        print(f"Existing CCOT result files: {len(_ccot_mat_files(args.out_dir))}")
        print(f"stdout log: {log_base.with_suffix('.log')}")
        print(f"stderr log: {log_base.with_suffix('.err.log')}")
        return 0

    if args.sequence_at_a_time:
        exit_code = _run_sequence_at_a_time(
            matlab_bin=args.matlab_bin,
            mode=args.mode,
            otb_root=args.otb_root,
            sequence_file=args.sequence_file,
            out_dir=args.out_dir,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        if exit_code != 0:
            return exit_code
        summary_path = summarize_results(args.out_dir, source_label, args.sequence_file)
        print(f"Existing CCOT result files: {len(_ccot_mat_files(args.out_dir))}")
        print(f"Wrote summary to {summary_path}")
        return 0

    exit_code = _run_foreground(command, REPO_ROOT)
    if exit_code != 0:
        print(f"MATLAB exited with code {exit_code}", file=sys.stderr)
        return exit_code

    summary_path = summarize_results(args.out_dir, source_label, args.sequence_file)
    print(f"Existing CCOT result files: {len(_ccot_mat_files(args.out_dir))}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
