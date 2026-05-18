"""Render PC MyTracker vs Pure CSRT challenge-representative plots.

This intentionally overwrites the original two comparison images:

- overall_result/challenge_representative_mytracker_vs_csrt.png
- overall_result/lasot_challenge_representative_mytracker_vs_csrt.png

OTB100 uses the PC/local MyTracker export, not the Jetson dual-acc run.
LaSOT uses the existing local full-tracker result bundle.
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
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "overall_result"


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    out_stem: str
    dataset_type: str
    my_result_dir: Path
    csrt_result_dir: Path
    sequences: tuple[dict[str, str], ...]


OTB_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT on Representative OTB Challenge Sequences",
    out_stem="challenge_representative_mytracker_vs_csrt",
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
    sequences=(
        {
            "challenge": "Occlusion",
            "challenge_code": "OCC",
            "title": "Occlusion (OCC)",
            "sequence": "Bolt",
        },
        {
            "challenge": "Illumination Variation",
            "challenge_code": "IV",
            "title": "Illumination Variation (IV)",
            "sequence": "Human8",
        },
        {
            "challenge": "Fast Motion",
            "challenge_code": "FM",
            "title": "Fast Motion (FM)",
            "sequence": "BlurCar3",
        },
    ),
)


LASOT_SPEC = DatasetSpec(
    title="CA-CSRT vs Pure CSRT on Representative LaSOT Challenge Sequences",
    out_stem="lasot_challenge_representative_mytracker_vs_csrt",
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
    sequences=(
        {
            "challenge": "Occlusion",
            "challenge_code": "POC/FOC",
            "title": "Occlusion (POC/FOC)",
            "sequence": "basketball-7",
        },
        {
            "challenge": "Illumination Variation",
            "challenge_code": "IV",
            "title": "Illumination Variation (IV)",
            "sequence": "shark-5",
        },
        {
            "challenge": "Fast Motion",
            "challenge_code": "FM",
            "title": "Fast Motion (FM)",
            "sequence": "airplane-13",
        },
    ),
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


def sequence_paths(dataset_type: str, sequence: str) -> tuple[Path, Path]:
    if dataset_type == "otb":
        seq_dir = REPO_ROOT / "otb" / "otb100" / sequence
        images = sorted((seq_dir / "img").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no OTB images found for {sequence}")
        return seq_dir / "groundtruth_rect.txt", images[0]

    if dataset_type == "lasot":
        category = sequence.split("-", 1)[0]
        seq_dir = REPO_ROOT / "ls" / "lasot" / category / sequence
        return seq_dir / "groundtruth.txt", seq_dir / "img" / "00000001.jpg"

    raise ValueError(f"unsupported dataset type: {dataset_type}")


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
    return out * 100.0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_dataset(spec: DatasetSpec) -> None:
    out_png = OUT_DIR / f"{spec.out_stem}.png"
    out_csv = OUT_DIR / f"{spec.out_stem}.csv"
    csv_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 13.625),
        dpi=160,
        gridspec_kw={"width_ratios": [1.38, 1.0], "hspace": 0.34, "wspace": 0.18},
    )
    fig.suptitle(spec.title, fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.945, bottom=0.055, wspace=0.18, hspace=0.42)

    for row_idx, seq_spec in enumerate(spec.sequences):
        sequence = seq_spec["sequence"]
        gt_path, image_path = sequence_paths(spec.dataset_type, sequence)
        my_path = spec.my_result_dir / f"{sequence}.txt"
        csrt_path = spec.csrt_result_dir / f"{sequence}.txt"

        gt = load_bbox(gt_path)
        my_bbox = load_bbox(my_path)
        csrt_bbox = load_bbox(csrt_path)
        my_iou = iou_per_frame(my_bbox, gt)
        csrt_iou = iou_per_frame(csrt_bbox, gt)
        my_auc = float(np.mean(my_iou))
        csrt_auc = float(np.mean(csrt_iou))
        delta = my_auc - csrt_auc

        ax_iou = axes[row_idx, 0]
        ax_iou.plot(np.arange(1, len(my_iou) + 1), my_iou, color="#1f77b4", linewidth=0.9, label="CA-CSRT")
        ax_iou.plot(np.arange(1, len(csrt_iou) + 1), csrt_iou, color="#ff7f0e", linewidth=0.9, label="Pure CSRT")
        ax_iou.set_xlim(1, max(len(my_iou), len(csrt_iou)))
        ax_iou.set_ylim(0, 100)
        ax_iou.set_xlabel("Frame", fontsize=8)
        ax_iou.set_ylabel("IoU vs Ground Truth (%)", fontsize=8)
        ax_iou.grid(True, linewidth=0.35, alpha=0.35)
        ax_iou.tick_params(labelsize=7)
        ax_iou.set_title(
            f"{seq_spec['title']} - {sequence}\n"
            f"AUC: CA-CSRT {my_auc:.1f} | CSRT {csrt_auc:.1f} | Delta {delta:+.1f}",
            fontsize=9,
            pad=6,
        )
        ax_iou.legend(loc="best", fontsize=7, framealpha=0.9)

        ax_traj = axes[row_idx, 1]
        ax_traj.imshow(Image.open(image_path))
        for bbox, color, label in (
            (my_bbox, "#1f77b4", "CA-CSRT trajectory"),
            (csrt_bbox, "#ff7f0e", "Pure CSRT trajectory"),
        ):
            center_x = bbox[:, 0] + bbox[:, 2] / 2.0
            center_y = bbox[:, 1] + bbox[:, 3] / 2.0
            valid = np.isfinite(center_x) & np.isfinite(center_y)
            ax_traj.plot(center_x[valid], center_y[valid], color=color, linewidth=0.95, label=label)
        ax_traj.set_title(f"Bounding-box Center Trajectory by Frame - {sequence}", fontsize=9, pad=6)
        ax_traj.set_xlabel("x pixel", fontsize=8)
        ax_traj.set_ylabel("y pixel", fontsize=8)
        ax_traj.tick_params(labelsize=7)
        if row_idx == 0:
            ax_traj.legend(loc="best", fontsize=7, framealpha=0.9)

        csv_rows.append(
            {
                "challenge": seq_spec["challenge"],
                "challenge_code": seq_spec["challenge_code"],
                "sequence": sequence,
                "frames_plotted": min(len(gt), len(my_bbox), len(csrt_bbox)),
                "my_auc_recomputed_from_pc_result": my_auc,
                "csrt_auc_recomputed": csrt_auc,
                "delta_auc_my_minus_csrt": delta,
                "mean_iou_my_recomputed": my_auc,
                "mean_iou_csrt_recomputed": csrt_auc,
                "my_result_file": rel(my_path),
                "csrt_result_file": rel(csrt_path),
                "groundtruth_file": rel(gt_path),
            }
        )

    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {rel(out_png)}")
    print(f"wrote {rel(out_csv)}")
    for row in csv_rows:
        print(
            f"{row['sequence']}: "
            f"MyTracker={row['my_auc_recomputed_from_pc_result']:.1f}, "
            f"CSRT={row['csrt_auc_recomputed']:.1f}, "
            f"Delta={row['delta_auc_my_minus_csrt']:+.1f}"
        )


def main() -> None:
    render_dataset(OTB_SPEC)
    render_dataset(LASOT_SPEC)


if __name__ == "__main__":
    main()
