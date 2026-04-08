import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.io import savemat


IGNORED_RESULT_SUFFIXES = (
    "_time",
    "_object_presence_scores",
)

OTB_SEQUENCE_NAME_MAP = {
    "Human4_2": "Human4-2",
    "Jogging_1": "Jogging-1",
    "Jogging_2": "Jogging-2",
    "Skating2_1": "Skating2-1",
    "Skating2_2": "Skating2-2",
}


def _is_sequence_result_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False

    stem = path.stem
    for suffix in IGNORED_RESULT_SUFFIXES:
        if stem.endswith(suffix):
            return False
    return True


def _read_txt_table(path: Path) -> np.ndarray:
    first_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break

    if not first_line:
        raise ValueError(f"Empty result file: {path}")

    delimiter = "," if "," in first_line else None
    data = np.loadtxt(path, dtype=np.float64, delimiter=delimiter)
    data = np.atleast_2d(data)
    return data


def _polygon_to_rect(poly: np.ndarray) -> np.ndarray:
    x = poly[:, 0::2]
    y = poly[:, 1::2]
    x1 = np.min(x, axis=1)
    y1 = np.min(y, axis=1)
    x2 = np.max(x, axis=1)
    y2 = np.max(y, axis=1)
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1, y1, w, h), axis=1)


def _load_boxes(path: Path) -> np.ndarray:
    data = _read_txt_table(path)

    if data.shape[1] == 4:
        boxes = data
    elif data.shape[1] == 8:
        boxes = _polygon_to_rect(data)
    else:
        raise ValueError(f"Unsupported bbox shape in {path}: {data.shape}")

    return boxes.astype(np.float64, copy=False)


def _load_fps(seq_result_file: Path) -> float:
    time_file = seq_result_file.with_name(f"{seq_result_file.stem}_time.txt")
    if not time_file.exists():
        return float("nan")

    time_data = _read_txt_table(time_file).reshape(-1)
    valid = time_data[np.isfinite(time_data) & (time_data > 0)]
    if valid.size == 0:
        return float("nan")

    return float(1.0 / np.mean(valid))


def _to_otb_sequence_name(seq_name: str) -> str:
    return OTB_SEQUENCE_NAME_MAP.get(seq_name, seq_name)


def _save_text_results(out_dir: Path, tracker_name: str, seq_name: str, boxes: np.ndarray) -> Path:
    tracker_dir = out_dir / tracker_name
    tracker_dir.mkdir(parents=True, exist_ok=True)
    seq_path = tracker_dir / f"{seq_name}.txt"
    np.savetxt(seq_path, boxes, delimiter=",", fmt="%.6f")
    return seq_path


def _save_classic_otb_mat(out_dir: Path, tracker_name: str, seq_name: str, boxes: np.ndarray, fps: float) -> Path:
    # Classic OTB toolkit expects results/OPE/<tracker>/<sequence>_*.mat with variable "res".
    tracker_dir = out_dir / "results" / "OPE" / tracker_name
    tracker_dir.mkdir(parents=True, exist_ok=True)
    mat_path = tracker_dir / f"{seq_name}_01.mat"

    payload = {"res": boxes}
    if not math.isnan(fps):
        payload["fps"] = np.array([[fps]], dtype=np.float64)
    savemat(mat_path, payload, do_compression=True)
    return mat_path


def _save_unified_otb_mat(out_dir: Path, tracker_name: str, seq_name: str, boxes: np.ndarray, fps: float) -> Path:
    # Unified toolkit expects results/OPE_OTB/<sequence>_<tracker>.mat with variable "results".
    mat_dir = out_dir / "results" / "OPE_OTB"
    mat_dir.mkdir(parents=True, exist_ok=True)
    mat_path = mat_dir / f"{seq_name}_{tracker_name}.mat"

    tracker_result = {
        "type": "rect",
        "res": boxes,
        "len": int(boxes.shape[0]),
        "annoBegin": 1,
        "startFrame": 1,
        "fps": float(0.0 if math.isnan(fps) else fps),
    }
    results = np.empty((1, 1), dtype=object)
    results[0, 0] = tracker_result
    savemat(mat_path, {"results": results}, do_compression=True)
    return mat_path


def _write_readme(output_dir: Path, tracker_name: str, source_dir: Path, num_sequences: int) -> None:
    readme_path = output_dir / "README_otb_toolkit.md"
    text = f"""# OTB MATLAB Export

Tracker exported: `{tracker_name}`
Source folder: `{source_dir}`
Sequences exported: `{num_sequences}`

## Exported folders
- `txt_results/{tracker_name}`: One bbox text file per sequence (`x,y,w,h`).
- `classic_otb/results/OPE/{tracker_name}`: `.mat` files for classic OTB toolkit (`res` variable).
- `unified_otb/results/OPE_OTB`: `.mat` files for unified toolkit (`results` variable).
- `manifest.csv`: Per-sequence mapping and FPS.

## Use with classic OTB toolkit (PAMI13 style)
1. Copy folder `classic_otb/results/OPE/{tracker_name}` into `<OTB_TOOLKIT_ROOT>/results/OPE/`.
2. Open `<OTB_TOOLKIT_ROOT>/configTrackers.m`.
3. Duplicate one existing tracker block and change:
   - tracker name -> `{tracker_name}`
   - result path -> `./results/OPE/{tracker_name}/`
4. Add `{tracker_name}` to the tracker list used by `perfPlot`.

## Use with unified OTB toolkit
1. Copy all `.mat` files from `unified_otb/results/OPE_OTB/` into `<UNIFIED_TOOLKIT_ROOT>/results/OPE_OTB/`.
2. Open `<UNIFIED_TOOLKIT_ROOT>/seqs/config_trackers.m`.
3. Add a tracker entry named `{tracker_name}` pointing to `results/OPE_OTB`.
4. Run evaluation/plot scripts with tracker list containing `{tracker_name}` and baselines (e.g. MDNet, CCOT, DeepSRDCF).
"""
    readme_path.write_text(text, encoding="utf-8")


def export_results(
    source_dir: Path,
    tracker_name: str,
    output_dir: Path,
    export_classic: bool,
    export_unified: bool,
) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    seq_files = sorted([p for p in source_dir.glob("*.txt") if _is_sequence_result_file(p)])
    if not seq_files:
        raise RuntimeError(f"No sequence result txt files found in: {source_dir}")

    txt_dir = output_dir / "txt_results"
    classic_dir = output_dir / "classic_otb"
    unified_dir = output_dir / "unified_otb"
    txt_dir.mkdir(parents=True, exist_ok=True)
    if export_classic:
        classic_dir.mkdir(parents=True, exist_ok=True)
    if export_unified:
        unified_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for seq_file in seq_files:
        seq_name = seq_file.stem
        otb_seq_name = _to_otb_sequence_name(seq_name)
        boxes = _load_boxes(seq_file)
        fps = _load_fps(seq_file)

        txt_path = _save_text_results(txt_dir, tracker_name, otb_seq_name, boxes)
        classic_path = ""
        unified_path = ""

        if export_classic:
            classic_path = str(_save_classic_otb_mat(classic_dir, tracker_name, otb_seq_name, boxes, fps))
        if export_unified:
            unified_path = str(_save_unified_otb_mat(unified_dir, tracker_name, otb_seq_name, boxes, fps))

        manifest_rows.append(
            {
                "sequence_src": seq_name,
                "sequence_otb": otb_seq_name,
                "frames": boxes.shape[0],
                "fps": "" if math.isnan(fps) else f"{fps:.6f}",
                "txt_path": str(txt_path),
                "classic_mat_path": classic_path,
                "unified_mat_path": unified_path,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_src",
                "sequence_otb",
                "frames",
                "fps",
                "txt_path",
                "classic_mat_path",
                "unified_mat_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    _write_readme(output_dir, tracker_name, source_dir, len(manifest_rows))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pytracking results to OTB MATLAB toolkit formats.")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing pytracking sequence result files (*.txt, *_time.txt).",
    )
    parser.add_argument(
        "--tracker-name",
        type=str,
        required=True,
        help="Tracker name to use in OTB toolkit (e.g. ToMP_plus2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Export output root directory.",
    )
    parser.add_argument(
        "--no-classic",
        action="store_true",
        help="Disable export for classic OTB toolkit format.",
    )
    parser.add_argument(
        "--no-unified",
        action="store_true",
        help="Disable export for unified OTB toolkit format.",
    )

    args = parser.parse_args()
    export_classic = not args.no_classic
    export_unified = not args.no_unified
    if not export_classic and not export_unified:
        raise ValueError("At least one export target must be enabled.")

    manifest_path = export_results(
        source_dir=Path(args.source_dir),
        tracker_name=args.tracker_name,
        output_dir=Path(args.output_dir),
        export_classic=export_classic,
        export_unified=export_unified,
    )
    print(f"Export completed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
