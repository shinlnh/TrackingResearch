"""Render OPE success plots for CA-CSRT vs Pure CSRT on the full datasets.

Produces:
- overall_result/otb100_full_mytracker_vs_csrt.png  (100 OTB100 sequences)
- overall_result/lasot_full_mytracker_vs_csrt.png   (40 LaSOT head-tail sequences)

OTB100 uses the PC/local MyTracker export. LaSOT uses the local full-tracker
result bundle.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "overall_result"

THRESHOLDS = np.linspace(0.0, 1.0, 21)
MY_COLOR = "#1f77b4"
CSRT_COLOR = "#ff7f0e"


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    out_stem: str
    dataset_type: str
    my_result_dir: Path
    csrt_result_dir: Path
    gt_root: Path


OTB_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT - OPE Success Plot on OTB100",
    out_stem="otb100_full_mytracker_vs_csrt",
    dataset_type="otb",
    my_result_dir=REPO_ROOT
    / "MyECOTracker"
    / "otb100result"
    / "otb_matlab_export"
    / "MyTracker_verified_otb936"
    / "txt_results"
    / "MyTracker",
    csrt_result_dir=REPO_ROOT
    / "OtherTracker"
    / "CSRT"
    / "otb100_results_full_20260326"
    / "txt_results"
    / "CSRT",
    gt_root=REPO_ROOT / "otb" / "otb100",
)


LASOT_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT - OPE Success Plot on LaSOT",
    out_stem="lasot_full_mytracker_vs_csrt",
    dataset_type="lasot",
    my_result_dir=REPO_ROOT
    / "overall_result"
    / "video"
    / "lasot"
    / "full_tracker"
    / "success_plot"
    / "tracking_results"
    / "MyTracker_tracking_result",
    csrt_result_dir=REPO_ROOT
    / "overall_result"
    / "video"
    / "lasot"
    / "full_tracker"
    / "success_plot"
    / "tracking_results"
    / "CSRT_tracking_result",
    gt_root=REPO_ROOT / "ls" / "lasot",
)


def load_bbox(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = [p for p in re.split(r"[\s,\t]+", line.strip()) if p]
            if len(parts) < 4:
                continue
            try:
                rows.append([float(v) for v in parts[:4]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no bbox rows found in {path}")
    return np.asarray(rows, dtype=float)


def gt_path_for(spec: DatasetSpec, sequence: str) -> Path:
    if spec.dataset_type == "otb":
        # Handle split OTB sequences like "Human4-2", "Jogging-1", "Skating2-2":
        # folder is the prefix, gt file is groundtruth_rect.<N>.txt.
        m = re.match(r"^(.+)-(\d+)$", sequence)
        if m:
            base, idx = m.group(1), m.group(2)
            split_folder = spec.gt_root / base
            split_gt = split_folder / f"groundtruth_rect.{idx}.txt"
            if split_gt.exists():
                return split_gt
        return spec.gt_root / sequence / "groundtruth_rect.txt"
    if spec.dataset_type == "lasot":
        category = sequence.split("-", 1)[0]
        return spec.gt_root / category / sequence / "groundtruth.txt"
    raise ValueError(f"unsupported dataset type: {spec.dataset_type}")


def iou_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    valid = (
        np.isfinite(pred).all(axis=1)
        & np.isfinite(gt).all(axis=1)
        & (pred[:, 2] > 0)
        & (pred[:, 3] > 0)
        & (gt[:, 2] > 0)
        & (gt[:, 3] > 0)
    )
    out = np.zeros(n, dtype=float)
    if not np.any(valid):
        return out

    p = pred[valid]
    g = gt[valid]
    px2 = p[:, 0] + p[:, 2]
    py2 = p[:, 1] + p[:, 3]
    gx2 = g[:, 0] + g[:, 2]
    gy2 = g[:, 1] + g[:, 3]

    ix1 = np.maximum(p[:, 0], g[:, 0])
    iy1 = np.maximum(p[:, 1], g[:, 1])
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter
    vals = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    out[valid] = vals
    return out


def success_curve(iou: np.ndarray) -> np.ndarray:
    return np.array([(iou >= t).mean() for t in THRESHOLDS], dtype=float)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_dataset(spec: DatasetSpec) -> None:
    sequences = sorted(p.stem for p in spec.my_result_dir.glob("*.txt"))
    csrt_sequences = {p.stem for p in spec.csrt_result_dir.glob("*.txt") if not p.stem.endswith("_time")}
    sequences = [s for s in sequences if s in csrt_sequences]

    if not sequences:
        raise RuntimeError(f"no overlapping sequences for {spec.out_stem}")

    my_curves: list[np.ndarray] = []
    csrt_curves: list[np.ndarray] = []
    csv_rows: list[dict[str, object]] = []

    for seq in sequences:
        gt = load_bbox(gt_path_for(spec, seq))
        my_bbox = load_bbox(spec.my_result_dir / f"{seq}.txt")
        csrt_bbox = load_bbox(spec.csrt_result_dir / f"{seq}.txt")
        my_iou = iou_per_frame(my_bbox, gt)
        csrt_iou = iou_per_frame(csrt_bbox, gt)
        my_curve = success_curve(my_iou)
        csrt_curve = success_curve(csrt_iou)
        my_curves.append(my_curve)
        csrt_curves.append(csrt_curve)
        csv_rows.append(
            {
                "sequence": seq,
                "frames": min(len(my_iou), len(csrt_iou)),
                "mytracker_auc": float(my_curve.mean()),
                "csrt_auc": float(csrt_curve.mean()),
                "delta_auc_my_minus_csrt": float(my_curve.mean() - csrt_curve.mean()),
            }
        )

    my_avg = np.mean(my_curves, axis=0)
    csrt_avg = np.mean(csrt_curves, axis=0)
    my_auc = float(my_avg.mean())
    csrt_auc = float(csrt_avg.mean())

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=160)
    ax.plot(THRESHOLDS, my_avg, color=MY_COLOR, linewidth=2.0,
            label=f"CA-CSRT [AUC {my_auc:.3f}]")
    ax.plot(THRESHOLDS, csrt_avg, color=CSRT_COLOR, linewidth=2.0, linestyle="--",
            label=f"Pure CSRT [AUC {csrt_auc:.3f}]")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Overlap threshold (IoU)", fontsize=11)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.set_title(
        f"{spec.title}\nDelta AUC (CA-CSRT - Pure CSRT) = {my_auc - csrt_auc:+.3f}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="lower left", fontsize=11, framealpha=0.92)

    fig.tight_layout()
    out_png = OUT_DIR / f"{spec.out_stem}.png"
    out_csv = OUT_DIR / f"{spec.out_stem}.csv"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {rel(out_png)}")
    print(f"wrote {rel(out_csv)}")
    print(f"  sequences: {len(sequences)}, CA-CSRT AUC={my_auc:.3f}, "
          f"Pure CSRT AUC={csrt_auc:.3f}, Delta={my_auc - csrt_auc:+.3f}")


def main() -> None:
    render_dataset(OTB_SPEC)
    render_dataset(LASOT_SPEC)


if __name__ == "__main__":
    main()
