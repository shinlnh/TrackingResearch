"""Plot OTB100 challenge-representative comparisons for three trackers.

This mirrors ``challenge_representative_mytracker_vs_csrt.png`` but compares
the PC/local MyTracker result, OSTrack, and STARK. Left panels show per-frame
IoU against ground truth, right panels show bounding-box center trajectories.
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
OUT_PNG = OUT_DIR / "otb100_challenge_representative_myeco_vs_ostrack_vs_stark.png"
OUT_CSV = OUT_DIR / "otb100_challenge_representative_myeco_vs_ostrack_vs_stark.csv"


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    label: str
    short_label: str
    color: str
    result_dir: Path


TRACKERS = [
    TrackerSpec(
        key="myeco",
        label="CA-CSRT",
        short_label="CA-CSRT",
        color="#1f77b4",
        result_dir=REPO_ROOT
        / "MyECOTracker"
        / "otb100result"
        / "otb_matlab_export"
        / "MyTracker_verified_otb936"
        / "txt_results"
        / "MyTracker",
    ),
    TrackerSpec(
        key="ostrack",
        label="OSTrack-384",
        short_label="OSTrack",
        color="#d62728",
        result_dir=REPO_ROOT
        / "OtherTracker"
        / "OSTrack"
        / "otb100_results"
        / "tracking_results"
        / "OSTrack",
    ),
    TrackerSpec(
        key="stark",
        label="STARK-ST101",
        short_label="STARK",
        color="#2ca02c",
        result_dir=REPO_ROOT
        / "OtherTracker"
        / "Stark"
        / "otb100_results"
        / "tracking_results"
        / "STARK",
    ),
]


SEQUENCES = [
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
]


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


def otb_paths(sequence: str) -> tuple[Path, Path]:
    seq_dir = REPO_ROOT / "otb" / "otb100" / sequence
    img_dir = seq_dir / "img"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no OTB images found in {img_dir}")
    return seq_dir / "groundtruth_rect.txt", images[0]


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


def plot() -> None:
    csv_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 13.625),
        dpi=160,
        gridspec_kw={"width_ratios": [1.38, 1.0], "hspace": 0.34, "wspace": 0.18},
    )
    fig.suptitle(
        "CA-CSRT vs OSTrack vs STARK on Representative OTB100 Challenge Sequences",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.945, bottom=0.055, wspace=0.18, hspace=0.42)

    for row_idx, spec in enumerate(SEQUENCES):
        sequence = spec["sequence"]
        gt_path, image_path = otb_paths(sequence)
        gt = load_bbox(gt_path)

        tracker_bboxes: dict[str, np.ndarray] = {}
        tracker_iou: dict[str, np.ndarray] = {}
        tracker_mean_iou: dict[str, float] = {}
        tracker_files: dict[str, Path] = {}

        for tracker in TRACKERS:
            result_path = tracker.result_dir / f"{sequence}.txt"
            bbox = load_bbox(result_path)
            tracker_bboxes[tracker.key] = bbox
            tracker_files[tracker.key] = result_path
            iou = iou_per_frame(bbox, gt)
            tracker_iou[tracker.key] = iou
            tracker_mean_iou[tracker.key] = float(np.mean(iou)) if len(iou) else float("nan")

        ax_iou = axes[row_idx, 0]
        for tracker in TRACKERS:
            y = tracker_iou[tracker.key]
            ax_iou.plot(
                np.arange(1, len(y) + 1),
                y,
                linewidth=0.9,
                color=tracker.color,
                label=tracker.label,
            )
        ax_iou.set_xlim(1, max(len(v) for v in tracker_iou.values()))
        ax_iou.set_ylim(0, 100)
        ax_iou.set_xlabel("Frame", fontsize=8)
        ax_iou.set_ylabel("IoU vs Ground Truth (%)", fontsize=8)
        ax_iou.grid(True, linewidth=0.35, alpha=0.35)
        ax_iou.tick_params(labelsize=7)

        auc_text = " | ".join(
            f"{tracker.short_label} {tracker_mean_iou[tracker.key]:.1f}" for tracker in TRACKERS
        )
        ax_iou.set_title(
            f"{spec['title']} - {sequence}\nAUC: {auc_text}",
            fontsize=9,
            pad=6,
        )
        ax_iou.legend(loc="best", fontsize=7, framealpha=0.9)

        ax_traj = axes[row_idx, 1]
        image = Image.open(image_path)
        ax_traj.imshow(image)
        for tracker in TRACKERS:
            bbox = tracker_bboxes[tracker.key]
            center_x = bbox[:, 0] + bbox[:, 2] / 2.0
            center_y = bbox[:, 1] + bbox[:, 3] / 2.0
            valid = np.isfinite(center_x) & np.isfinite(center_y)
            ax_traj.plot(
                center_x[valid],
                center_y[valid],
                linewidth=0.95,
                color=tracker.color,
                label=f"{tracker.label} trajectory",
            )
        ax_traj.set_title(
            f"Bounding-box Center Trajectory by Frame - {sequence}",
            fontsize=9,
            pad=6,
        )
        ax_traj.set_xlabel("x pixel", fontsize=8)
        ax_traj.set_ylabel("y pixel", fontsize=8)
        ax_traj.tick_params(labelsize=7)
        if row_idx == 0:
            ax_traj.legend(loc="best", fontsize=7, framealpha=0.9)

        csv_rows.append(
            {
                "challenge": spec["challenge"],
                "challenge_code": spec["challenge_code"],
                "sequence": sequence,
                "frames_plotted": min(len(gt), *(len(v) for v in tracker_bboxes.values())),
                "mytracker_auc_recomputed_from_pc_result": tracker_mean_iou["myeco"],
                "ostrack_auc_recomputed": tracker_mean_iou["ostrack"],
                "stark_auc_recomputed": tracker_mean_iou["stark"],
                "mean_iou_mytracker_recomputed": tracker_mean_iou["myeco"],
                "mean_iou_ostrack_recomputed": tracker_mean_iou["ostrack"],
                "mean_iou_stark_recomputed": tracker_mean_iou["stark"],
                "mytracker_result_file": rel(tracker_files["myeco"]),
                "ostrack_result_file": rel(tracker_files["ostrack"]),
                "stark_result_file": rel(tracker_files["stark"]),
                "groundtruth_file": rel(gt_path),
            }
        )

    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {rel(OUT_PNG)}")
    print(f"wrote {rel(OUT_CSV)}")
    for row in csv_rows:
        print(
            f"{row['sequence']}: "
            f"MyTracker={row['mytracker_auc_recomputed_from_pc_result']:.1f}, "
            f"OSTrack={row['ostrack_auc_recomputed']:.1f}, "
            f"STARK={row['stark_auc_recomputed']:.1f}"
        )


if __name__ == "__main__":
    plot()
