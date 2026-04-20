#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MyECOTracker on the hardest OTB scale-variation sequences."
    )
    parser.add_argument("--tracker-name", default="eco")
    parser.add_argument("--param", default="verified_otb936")
    parser.add_argument("--top-n", type=int, default=10, help="How many SV sequences to run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "myecotracker_otb_sv_top10",
    )
    parser.add_argument(
        "--save-preview-frames",
        type=int,
        default=3,
        help="How many evenly sampled overlay frames to save per sequence.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent


def setup_pytracking(root: Path) -> None:
    pytracking_dir = root / "MyECOTracker" / "pytracking"
    if str(pytracking_dir) not in sys.path:
        sys.path.insert(0, str(pytracking_dir))


def load_otb_sv_sequences(root: Path):
    setup_pytracking(root)
    from pytracking.evaluation import get_dataset

    attrs_path = (
        root
        / "MyECOTracker"
        / "pytracking"
        / "pytracking"
        / "evaluation"
        / "dataset_attribute_specs"
        / "otb_attributes.json"
    )
    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    otb = get_dataset("otb")

    rows = []
    for seq in otb:
        if "SV" not in attrs.get(seq.name, []):
            continue
        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        valid = gt[:, 2] > 0
        valid &= gt[:, 3] > 0
        gt = gt[valid]
        if gt.size == 0:
            continue
        areas = gt[:, 2] * gt[:, 3]
        ratio = float(np.max(areas) / np.min(areas))
        rows.append(
            {
                "sequence": seq,
                "scale_area_ratio": ratio,
                "scale_linear_ratio": float(np.sqrt(ratio)),
                "frames": int(len(seq.frames)),
            }
        )

    rows.sort(key=lambda r: r["scale_area_ratio"], reverse=True)
    return rows


def calc_metrics(pred: np.ndarray, gt: np.ndarray, dataset_name: str):
    from pytracking.analysis.extract_results import calc_seq_err_robust

    pred_t = torch.tensor(pred, dtype=torch.float64)
    gt_t = torch.tensor(gt, dtype=torch.float64)
    err_overlap, err_center, err_center_norm, valid = calc_seq_err_robust(pred_t, gt_t, dataset_name)
    seq_length = gt_t.shape[0]

    return {
        "avg_overlap": float(err_overlap[valid].mean().item()),
        "success50": float((err_overlap > 0.50).sum().item() / seq_length),
        "success75": float((err_overlap > 0.75).sum().item() / seq_length),
        "precision20": float((err_center <= 20.0).sum().item() / seq_length),
        "norm_precision20": float((err_center_norm <= 0.20).sum().item() / seq_length),
        "err_overlap": err_overlap.numpy(),
        "err_center": err_center.numpy(),
    }


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        image,
        label,
        (max(0, x), max(16, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def save_preview_frames(
    seq_name: str,
    frame_paths: list[str],
    gt: np.ndarray,
    pred: np.ndarray,
    iou: np.ndarray,
    ce: np.ndarray,
    out_dir: Path,
    sample_count: int,
) -> None:
    seq_dir = out_dir / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frame_paths) - 1, num=min(sample_count, len(frame_paths)), dtype=int)

    for idx in indices:
        image = cv2.imread(frame_paths[idx])
        if image is None:
            raise RuntimeError(f"Failed to read {frame_paths[idx]}")
        draw_box(image, gt[idx], (0, 255, 0), "GT")
        draw_box(image, pred[idx], (0, 0, 255), "ECO")
        cv2.putText(
            image,
            f"{seq_name} frame={idx} IoU={float(iou[idx]):.3f} CE={float(ce[idx]):.1f}px",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(seq_dir / Path(frame_paths[idx]).name), image)


def main() -> int:
    args = parse_args()
    root = project_root()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_pytracking(root)
    from pytracking.evaluation.tracker import Tracker

    candidates = load_otb_sv_sequences(root)[: args.top_n]
    tracker = Tracker(args.tracker_name, args.param)

    summary_rows = []
    for idx, item in enumerate(candidates, start=1):
        seq = item["sequence"]
        print(
            f"[{idx}/{len(candidates)}] {seq.name} "
            f"scale_linear_ratio={item['scale_linear_ratio']:.3f} frames={item['frames']}"
        )
        output = tracker.run_sequence(seq, debug=0, visdom_info={"use_visdom": False})
        pred = np.asarray(output["target_bbox"], dtype=np.float64)
        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        times = np.asarray(output["time"], dtype=np.float64)

        metrics = calc_metrics(pred, gt, seq.dataset)
        fps = float(len(times) / np.sum(times)) if np.sum(times) > 0 else float("nan")

        seq_result_dir = out_dir / "per_sequence"
        seq_result_dir.mkdir(exist_ok=True)
        with (seq_result_dir / f"{seq.name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_index",
                    "frame_path",
                    "gt_x",
                    "gt_y",
                    "gt_w",
                    "gt_h",
                    "pred_x",
                    "pred_y",
                    "pred_w",
                    "pred_h",
                    "iou",
                    "center_error_px",
                    "time_s",
                ]
            )
            for frame_idx, frame_path in enumerate(seq.frames):
                writer.writerow(
                    [
                        frame_idx,
                        frame_path,
                        *gt[frame_idx].tolist(),
                        *pred[frame_idx].tolist(),
                        float(metrics["err_overlap"][frame_idx]),
                        float(metrics["err_center"][frame_idx]),
                        float(times[frame_idx]),
                    ]
                )

        save_preview_frames(
            seq_name=seq.name,
            frame_paths=list(seq.frames),
            gt=gt,
            pred=pred,
            iou=metrics["err_overlap"],
            ce=metrics["err_center"],
            out_dir=out_dir / "preview_frames",
            sample_count=args.save_preview_frames,
        )

        summary_rows.append(
            {
                "rank_by_scale": idx,
                "sequence": seq.name,
                "frames": len(seq.frames),
                "scale_area_ratio": item["scale_area_ratio"],
                "scale_linear_ratio": item["scale_linear_ratio"],
                "AUC_mean": metrics["avg_overlap"] * 100.0,
                "Success50": metrics["success50"] * 100.0,
                "Success75": metrics["success75"] * 100.0,
                "Precision20": metrics["precision20"] * 100.0,
                "NormPrecision20": metrics["norm_precision20"] * 100.0,
                "FPS": fps,
            }
        )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_by_scale",
                "sequence",
                "frames",
                "scale_area_ratio",
                "scale_linear_ratio",
                "AUC_mean",
                "Success50",
                "Success75",
                "Precision20",
                "NormPrecision20",
                "FPS",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if summary_rows:
        aucs = np.asarray([row["AUC_mean"] for row in summary_rows], dtype=np.float64)
        prec20 = np.asarray([row["Precision20"] for row in summary_rows], dtype=np.float64)
        fps = np.asarray([row["FPS"] for row in summary_rows], dtype=np.float64)
        with (out_dir / "aggregate.txt").open("w", encoding="utf-8") as f:
            f.write(f"tracker={args.tracker_name}\n")
            f.write(f"param={args.param}\n")
            f.write(f"sequences={len(summary_rows)}\n")
            f.write(f"AUC_mean={float(np.mean(aucs)):.6f}\n")
            f.write(f"Precision20_mean={float(np.mean(prec20)):.6f}\n")
            f.write(f"FPS_mean={float(np.mean(fps)):.6f}\n")
            worst = min(summary_rows, key=lambda r: r["AUC_mean"])
            best = max(summary_rows, key=lambda r: r["AUC_mean"])
            f.write(f"best_sequence={best['sequence']}:{best['AUC_mean']:.6f}\n")
            f.write(f"worst_sequence={worst['sequence']}:{worst['AUC_mean']:.6f}\n")

    print(f"summary_csv={summary_path}")
    print(f"aggregate_txt={out_dir / 'aggregate.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
