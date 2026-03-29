from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def _resolve_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    pymdnet_root = repo_root / "OtherTracker" / "PyMDNet"
    pytracking_root = repo_root / "MyECOTracker" / "pytracking"
    if not pymdnet_root.exists():
        raise FileNotFoundError(f"PyMDNet repository not found at {pymdnet_root}")
    if not pytracking_root.exists():
        raise FileNotFoundError(f"pytracking workspace not found at {pytracking_root}")
    return repo_root, pymdnet_root, pytracking_root


def _bootstrap_pytracking(pytracking_root: Path):
    root_str = str(pytracking_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from pytracking.analysis.extract_results import extract_results
    from pytracking.analysis.plot_results import get_auc_curve, get_prec_curve
    from pytracking.evaluation import get_dataset

    return extract_results, get_auc_curve, get_prec_curve, get_dataset


def _bootstrap_pymdnet(pymdnet_root: Path):
    cwd_prev = Path.cwd()
    os.chdir(pymdnet_root)
    try:
        for path in (pymdnet_root / "tracking", pymdnet_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        import run_tracker as pymdnet_run_tracker
    finally:
        os.chdir(cwd_prev)

    return pymdnet_run_tracker


def _load_dataset(get_dataset, sequence_file: Path | None):
    dataset = list(get_dataset("lasot"))
    if sequence_file is None:
        return dataset

    raw_names = [
        line.strip()
        for line in sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_name = {seq.name: seq for seq in dataset}

    filtered = []
    missing = []
    for name in raw_names:
        seq = by_name.get(name)
        if seq is None:
            missing.append(name)
        else:
            filtered.append(seq)

    if missing:
        raise ValueError(f"Unknown LaSOT sequences: {', '.join(missing)}")

    return filtered


def _filter_single_sequence(dataset, sequence_name: str | None):
    if not sequence_name:
        return dataset

    for seq in dataset:
        if seq.name == sequence_name:
            return [seq]

    raise ValueError(f"Unknown LaSOT sequence: {sequence_name}")


def _save_txt_result(path: Path, result_bb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, result_bb, fmt="%.6f", delimiter=",")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sequence", "frames", "fps", "txt_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _load_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["sequence"]] = {
                "sequence": row["sequence"],
                "frames": int(row["frames"]),
                "fps": float(row["fps"]),
                "txt_path": row["txt_path"],
            }
    return rows


def _scope_name(args: argparse.Namespace) -> str:
    if args.sequence:
        return args.sequence
    if args.sequence_file is None:
        return "testset280"
    return args.sequence_file.stem


def _valid_manifest_row(row: dict[str, object] | None, txt_path: Path) -> bool:
    if row is None or not txt_path.exists():
        return False
    try:
        fps = float(row["fps"])
    except Exception:
        return False
    return math.isfinite(fps) and fps > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PyMDNet on the LaSOT test set and save results under OtherTracker/MDNet.")
    parser.add_argument("--sequence", help="Optional single LaSOT sequence to run.")
    parser.add_argument("--sequence-file", type=Path, help="Optional file listing LaSOT sequence names to run.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults under OtherTracker/MDNet.")
    parser.add_argument("--display", action="store_true", help="Show tracker visualization.")
    parser.add_argument("--model-path", type=Path, help="Override PyMDNet checkpoint path.")
    parser.add_argument("--display-name", default="MDNet")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing txt result + manifest rows when present.")
    args = parser.parse_args()

    repo_root, pymdnet_root, pytracking_root = _resolve_paths()
    extract_results, get_auc_curve, get_prec_curve, get_dataset = _bootstrap_pytracking(pytracking_root)
    pymdnet_run_tracker = _bootstrap_pymdnet(pymdnet_root)

    np.random.seed(0)
    torch.manual_seed(0)

    dataset = _load_dataset(get_dataset, args.sequence_file)
    dataset = _filter_single_sequence(dataset, args.sequence)

    scope = _scope_name(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "OtherTracker" / "MDNet" / f"lasot_pymdnet_vototb_{scope}"
    output_dir = output_dir.resolve()
    txt_dir = output_dir / "txt_results" / "MDNet"
    log_dir = output_dir / "sequence_logs"
    manifest_path = output_dir / "manifest.csv"
    existing_manifest = _load_manifest(manifest_path) if args.skip_existing else {}

    model_path = args.model_path.resolve() if args.model_path else (pymdnet_root / "models" / "mdnet_vot-otb.pth").resolve()
    pymdnet_run_tracker.opts["model_path"] = str(model_path)
    pymdnet_run_tracker.opts["use_gpu"] = torch.cuda.is_available()

    manifest_rows: list[dict[str, object]] = []
    total_frames = 0
    total_time = 0.0
    fps_values: list[float] = []

    cwd_prev = Path.cwd()
    os.chdir(pymdnet_root)
    try:
        for seq in dataset:
            txt_path = txt_dir / f"{seq.name}.txt"
            existing_row = existing_manifest.get(seq.name)
            if args.skip_existing and _valid_manifest_row(existing_row, txt_path):
                frames = int(existing_row["frames"])
                fps = float(existing_row["fps"])
                print(f"Skipping {seq.name} and reusing existing result ({fps:.3f} FPS)", flush=True)
                row = {
                    "sequence": seq.name,
                    "frames": frames,
                    "fps": f"{fps:.6f}",
                    "txt_path": str(txt_path),
                }
            else:
                print(f"Running PyMDNet on {seq.name} ({len(seq.frames)} frames)", flush=True)
                seq_log_path = log_dir / f"{seq.name}.log"
                seq_log_path.parent.mkdir(parents=True, exist_ok=True)
                with seq_log_path.open("w", encoding="utf-8") as seq_log_fh, contextlib.redirect_stdout(seq_log_fh):
                    result, result_bb, fps = pymdnet_run_tracker.run_mdnet(
                        seq.frames,
                        seq.ground_truth_rect[0].tolist(),
                        gt=seq.ground_truth_rect,
                        savefig_dir="",
                        display=args.display,
                    )
                _ = result

                _save_txt_result(txt_path, result_bb)

                row = {
                    "sequence": seq.name,
                    "frames": len(seq.frames),
                    "fps": f"{float(fps):.6f}",
                    "txt_path": str(txt_path),
                }
                print(f"Completed {seq.name}: {float(fps):.3f} FPS", flush=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            frames = int(row["frames"])
            fps = float(row["fps"])
            total_frames += frames
            fps_values.append(fps)
            total_time += frames / fps
            manifest_rows.append(row)
            _write_manifest(manifest_path, manifest_rows)
    finally:
        os.chdir(cwd_prev)

    tracker_stub = SimpleNamespace(
        name="MDNet",
        parameter_name="pymdnet_vototb",
        run_id=None,
        display_name=args.display_name,
        results_dir=str(txt_dir),
    )
    report_name = f"mdnet_pymdnet_lasot_{scope}"
    eval_data = extract_results([tracker_stub], dataset, report_name, skip_missing_seq=False, verbose=False)

    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    auc_curve, auc = get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
    _, precision = get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

    fps_values_np = np.asarray(fps_values, dtype=np.float64)
    summary = {
        "tracker": args.display_name,
        "parameter": "pymdnet_vototb",
        "scope": scope,
        "valid_sequences": int(valid.sum().item()),
        "AUC": float(auc[0]),
        "Precision": float(precision[0]),
        "Success50": float(auc_curve[0, 10]),
        "FPS_avg_seq": float(np.nanmean(fps_values_np)),
        "FPS_median_seq": float(np.nanmedian(fps_values_np)),
        "FPS_weighted_by_frames": float(total_frames / total_time) if total_time > 0 else float("nan"),
        "total_frames": int(total_frames),
        "total_time_sec": float(total_time),
        "model_path": str(model_path),
    }
    summary_path = output_dir / "summary.csv"
    _write_summary(summary_path, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
