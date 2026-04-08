from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np


PLOT_BIN_GAP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    return parser.parse_args()


def load_text_array(path: Path, delimiter: str = ",") -> np.ndarray:
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    return np.atleast_1d(data)


def load_bbox_array(path: Path) -> np.ndarray:
    for delimiter in (",", "\t", None):
        try:
            if delimiter is None:
                data = np.loadtxt(path, dtype=np.float64)
            else:
                data = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
            data = np.asarray(data, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] == 4:
                return data
        except ValueError:
            continue
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = [tok for tok in re.split(r"[\s,;]+", line) if tok]
        if len(tokens) < 4:
            continue
        row = [parse_maybe_complex_float(tok) for tok in tokens[:4]]
        rows.append(row)
    if rows:
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] == 4:
            return data
    raise ValueError(f"Could not parse bbox file: {path}")


def parse_maybe_complex_float(token: str) -> float:
    token = token.strip().strip("[](){}")
    if not token:
        raise ValueError("Empty numeric token")
    if "nan" in token.lower():
        return float("nan")
    token = token.replace("I", "i").replace("J", "j").replace("i", "j")
    try:
        return float(token)
    except ValueError:
        return float(complex(token).real)


def sanitize_bbox_array(bboxes: np.ndarray) -> np.ndarray:
    bboxes = np.array(bboxes, dtype=np.float64, copy=True)

    if np.isnan(bboxes).any() or np.isinf(bboxes).any():
        for row_idx in range(bboxes.shape[0]):
            invalid = ~np.isfinite(bboxes[row_idx])
            if not np.any(invalid):
                continue
            if row_idx > 0:
                bboxes[row_idx, invalid] = bboxes[row_idx - 1, invalid]
            if np.any(~np.isfinite(bboxes[row_idx])):
                bboxes[row_idx, ~np.isfinite(bboxes[row_idx])] = 0.0

    neg_w = bboxes[:, 2] < 0.0
    if np.any(neg_w):
        bboxes[neg_w, 0] += bboxes[neg_w, 2]
        bboxes[neg_w, 2] = -bboxes[neg_w, 2]

    neg_h = bboxes[:, 3] < 0.0
    if np.any(neg_h):
        bboxes[neg_h, 1] += bboxes[neg_h, 3]
        bboxes[neg_h, 3] = -bboxes[neg_h, 3]

    return bboxes


def calc_err_center(pred_bb: np.ndarray, anno_bb: np.ndarray, normalized: bool = False) -> np.ndarray:
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        denom = anno_bb[:, 2:].copy()
        denom[denom <= 0.0] = np.nan
        pred_center = pred_center / denom
        anno_center = anno_center / denom

    return np.sqrt(np.sum((pred_center - anno_center) ** 2, axis=1))


def calc_iou_overlap(pred_bb: np.ndarray, anno_bb: np.ndarray) -> np.ndarray:
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = np.clip(br - tl + 1.0, 0.0, None)

    intersection = sz[:, 0] * sz[:, 1]
    union = pred_bb[:, 2] * pred_bb[:, 3] + anno_bb[:, 2] * anno_bb[:, 3] - intersection
    overlap = np.zeros_like(intersection, dtype=np.float64)
    valid_union = union > 0.0
    overlap[valid_union] = intersection[valid_union] / union[valid_union]
    return overlap


def calc_seq_err_robust(
    pred_bb: np.ndarray,
    anno_bb: np.ndarray,
    target_visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_bb = sanitize_bbox_array(pred_bb)
    anno_bb = np.array(anno_bb, dtype=np.float64, copy=False)
    target_visible = np.asarray(target_visible, dtype=bool).reshape(-1)

    if np.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise ValueError("Invalid tracker results")

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if pred_bb.shape[0] > anno_bb.shape[0]:
            pred_bb = pred_bb[: anno_bb.shape[0], :]
        else:
            raise ValueError("Tracker prediction shorter than GT")

    pred_bb[0, :] = anno_bb[0, :]

    valid = np.all(anno_bb > 0.0, axis=1) & target_visible

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    err_center[~valid] = np.inf
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    err_center_normalized[~target_visible] = np.inf
    err_center[~target_visible] = np.inf

    if np.isnan(err_overlap).any():
        raise ValueError("NaNs in overlap computation")

    return err_overlap, err_center, err_center_normalized, valid


def evaluate_sequence(
    repo_root: Path,
    results_dir: Path,
    sequence_name: str,
    threshold_set_overlap: np.ndarray,
    threshold_set_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    class_name = sequence_name.split("-")[0]
    seq_dir = repo_root / "ls" / "lasot" / class_name / sequence_name

    anno_bb = load_bbox_array(seq_dir / "groundtruth.txt")
    full_occlusion = load_text_array(seq_dir / "full_occlusion.txt").reshape(-1)
    out_of_view = load_text_array(seq_dir / "out_of_view.txt").reshape(-1)
    target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

    pred_bb = load_bbox_array(results_dir / f"{sequence_name}.txt")
    err_overlap, err_center, _, _ = calc_seq_err_robust(pred_bb, anno_bb, target_visible)

    seq_length = int(anno_bb.shape[0])
    overlap_curve = (err_overlap[:, None] > threshold_set_overlap[None, :]).sum(axis=0).astype(np.float64) / seq_length
    center_curve = (err_center[:, None] <= threshold_set_center[None, :]).sum(axis=0).astype(np.float64) / seq_length

    time_path = results_dir / f"{sequence_name}_time.txt"
    times = load_text_array(time_path, delimiter="\t").reshape(-1).astype(np.float64)
    seq_time = float(np.sum(times))

    return overlap_curve, center_curve, seq_length, seq_time


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    results_dir = args.results_dir.resolve()

    raw_names = [
        line.strip()
        for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    threshold_set_overlap = np.arange(0.0, 1.0 + PLOT_BIN_GAP, PLOT_BIN_GAP, dtype=np.float64)
    threshold_set_center = np.arange(0, 51, dtype=np.float64)

    overlap_curves = []
    center_curves = []
    fps_values = []
    frames_total = 0
    total_time = 0.0

    for sequence_name in raw_names:
        overlap_curve, center_curve, seq_length, seq_time = evaluate_sequence(
            repo_root,
            results_dir,
            sequence_name,
            threshold_set_overlap,
            threshold_set_center,
        )
        overlap_curves.append(overlap_curve)
        center_curves.append(center_curve)
        frames_total += seq_length
        total_time += seq_time
        fps_values.append(seq_length / seq_time if seq_time > 0 else np.nan)

    overlap_curves_arr = np.asarray(overlap_curves, dtype=np.float64)
    center_curves_arr = np.asarray(center_curves, dtype=np.float64)
    fps_values_arr = np.asarray(fps_values, dtype=np.float64)

    auc_curve = overlap_curves_arr.mean(axis=0) * 100.0
    prec_curve = center_curves_arr.mean(axis=0) * 100.0

    summary = {
        "tracker": args.tracker_name,
        "scope": "headtail40",
        "valid_sequences": len(raw_names),
        "AUC": float(auc_curve.mean()),
        "Precision": float(prec_curve[20]),
        "Success50": float(auc_curve[10]),
        "FPS_avg_seq": float(np.nanmean(fps_values_arr)),
        "FPS_median_seq": float(np.nanmedian(fps_values_arr)),
        "FPS_weighted_by_frames": float(frames_total / total_time) if total_time > 0 else float("nan"),
        "total_frames": int(frames_total),
        "total_time_sec": float(total_time),
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
