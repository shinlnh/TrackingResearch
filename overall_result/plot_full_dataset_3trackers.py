"""Render OPE success plots for CA-CSRT vs OSTrack vs STARK on the full datasets.

Produces:
- overall_result/otb100_full_mytracker_vs_ostrack_vs_stark.png  (100 OTB100 seqs)
- overall_result/lasot_full_mytracker_vs_ostrack_vs_stark.png   (40 LaSOT seqs)
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


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    label: str
    color: str
    linestyle: str
    result_dir: Path


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    out_stem: str
    dataset_type: str
    gt_root: Path
    trackers: tuple[TrackerSpec, ...]


OTB_SPEC = DatasetSpec(
    title="CA-CSRT vs OSTrack vs STARK - OPE Success Plot on OTB100",
    out_stem="otb100_full_mytracker_vs_ostrack_vs_stark",
    dataset_type="otb",
    gt_root=REPO_ROOT / "otb" / "otb100",
    trackers=(
        TrackerSpec(
            key="myeco",
            label="CA-CSRT",
            color="#1f77b4",
            linestyle="-",
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
            color="#d62728",
            linestyle="--",
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
            color="#2ca02c",
            linestyle="-.",
            result_dir=REPO_ROOT
            / "OtherTracker"
            / "Stark"
            / "otb100_results"
            / "tracking_results"
            / "STARK",
        ),
    ),
)


LASOT_SPEC = DatasetSpec(
    title="CA-CSRT vs OSTrack vs STARK - OPE Success Plot on LaSOT",
    out_stem="lasot_full_mytracker_vs_ostrack_vs_stark",
    dataset_type="lasot",
    gt_root=REPO_ROOT / "ls" / "lasot",
    trackers=(
        TrackerSpec(
            key="myeco",
            label="CA-CSRT",
            color="#1f77b4",
            linestyle="-",
            result_dir=REPO_ROOT
            / "overall_result"
            / "video"
            / "lasot"
            / "full_tracker"
            / "success_plot"
            / "tracking_results"
            / "MyTracker_tracking_result",
        ),
        TrackerSpec(
            key="ostrack",
            label="OSTrack-384",
            color="#d62728",
            linestyle="--",
            result_dir=REPO_ROOT
            / "OtherTracker"
            / "lasot"
            / "lasot936"
            / "OSTrack"
            / "tracking_results"
            / "OSTrack",
        ),
        TrackerSpec(
            key="stark",
            label="STARK-ST101",
            color="#2ca02c",
            linestyle="-.",
            result_dir=REPO_ROOT
            / "OtherTracker"
            / "lasot"
            / "lasot936"
            / "STARK"
            / "tracking_results"
            / "STARK",
        ),
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


def gt_path_for(spec: DatasetSpec, sequence: str) -> Path:
    if spec.dataset_type == "otb":
        m = re.match(r"^(.+)-(\d+)$", sequence)
        if m:
            base, idx = m.group(1), m.group(2)
            split_gt = spec.gt_root / base / f"groundtruth_rect.{idx}.txt"
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


def _norm(name: str) -> str:
    # OTB split sequences appear as both "Human4-2" and "Human4_2" across
    # different tracker output dirs; normalize for matching.
    return name.replace("_", "-")


def render_dataset(spec: DatasetSpec) -> None:
    # Find sequences common to all trackers using normalized names.
    per_tracker_files: list[dict[str, Path]] = []
    for tr in spec.trackers:
        m: dict[str, Path] = {}
        for p in tr.result_dir.glob("*.txt"):
            if p.stem.endswith("_time"):
                continue
            m[_norm(p.stem)] = p
        per_tracker_files.append(m)

    common = sorted(set.intersection(*(set(m.keys()) for m in per_tracker_files)))
    if not common:
        raise RuntimeError(f"no overlapping sequences for {spec.out_stem}")

    per_tracker_curves: dict[str, list[np.ndarray]] = {tr.key: [] for tr in spec.trackers}
    csv_rows: list[dict[str, object]] = []

    for seq in common:
        gt = load_bbox(gt_path_for(spec, seq))
        per_seq_auc: dict[str, float] = {}
        frames_min = len(gt)
        for tr_idx, tr in enumerate(spec.trackers):
            bbox = load_bbox(per_tracker_files[tr_idx][seq])
            iou = iou_per_frame(bbox, gt)
            curve = success_curve(iou)
            per_tracker_curves[tr.key].append(curve)
            per_seq_auc[tr.key] = float(curve.mean())
            frames_min = min(frames_min, len(bbox))
        row: dict[str, object] = {"sequence": seq, "frames": frames_min}
        for tr in spec.trackers:
            row[f"{tr.key}_auc"] = per_seq_auc[tr.key]
        csv_rows.append(row)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=160)
    auc_by_key: dict[str, float] = {}
    for tr in spec.trackers:
        curve_avg = np.mean(per_tracker_curves[tr.key], axis=0)
        auc = float(curve_avg.mean())
        auc_by_key[tr.key] = auc
        ax.plot(
            THRESHOLDS,
            curve_avg,
            color=tr.color,
            linestyle=tr.linestyle,
            linewidth=2.0,
            label=f"{tr.label} [AUC {auc:.3f}]",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Overlap threshold (IoU)", fontsize=11)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.set_title(spec.title, fontsize=11, fontweight="bold", pad=10)
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
    aucs_str = ", ".join(f"{tr.label} AUC={auc_by_key[tr.key]:.3f}" for tr in spec.trackers)
    print(f"  sequences: {len(common)}, {aucs_str}")


def main() -> None:
    render_dataset(OTB_SPEC)
    render_dataset(LASOT_SPEC)


if __name__ == "__main__":
    main()
