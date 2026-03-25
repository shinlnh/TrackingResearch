from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OtherTracker.tools.otb_sequences import load_otb_sequences


def _load_boxes(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(v) for v in line.replace("\t", ",").split(",")[:4]])
    return np.asarray(rows, dtype=np.float64)


def _bbox_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    px1 = pred[:, 0]
    py1 = pred[:, 1]
    px2 = px1 + pred[:, 2]
    py2 = py1 + pred[:, 3]

    gx1 = gt[:, 0]
    gy1 = gt[:, 1]
    gx2 = gx1 + gt[:, 2]
    gy2 = gy1 + gt[:, 3]

    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return inter / np.clip(union, 1e-6, None)


def _center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pcx = pred[:, 0] + pred[:, 2] / 2.0
    pcy = pred[:, 1] + pred[:, 3] / 2.0
    gcx = gt[:, 0] + gt[:, 2] / 2.0
    gcy = gt[:, 1] + gt[:, 3] / 2.0
    return np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--otb-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    sequences = load_otb_sequences(args.otb_root, args.sequence_file)
    txt_dir = args.results_dir / "txt_results" / "CNN-SVM-Approx"

    thresholds = np.linspace(0.0, 1.0, 21)
    auc_values = []
    precision_values = []
    rows = []

    for sequence in sequences:
        pred_path = txt_dir / f"{sequence.name}.txt"
        if not pred_path.exists():
            continue
        pred = _load_boxes(pred_path)
        gt = np.asarray(sequence.groundtruth_rects, dtype=np.float64)
        count = min(len(pred), len(gt))
        pred = pred[:count]
        gt = gt[:count]

        ious = _bbox_iou(pred, gt)
        errors = _center_error(pred, gt)
        success_curve = [(ious >= thr).mean() for thr in thresholds]
        auc = float(np.mean(success_curve))
        precision20 = float((errors <= 20.0).mean())
        auc_values.append(auc)
        precision_values.append(precision20)
        rows.append(
            {
                "sequence": sequence.name,
                "frames": count,
                "auc": auc,
                "precision20": precision20,
                "success50": float((ious >= 0.5).mean()),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frames", "auc", "precision20", "success50"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "sequence": "OVERALL",
                "frames": sum(int(row["frames"]) for row in rows),
                "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
                "precision20": float(np.mean(precision_values)) if precision_values else float("nan"),
                "success50": float(np.mean([row["success50"] for row in rows])) if rows else float("nan"),
            }
        )
    print(f"Wrote evaluation to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
