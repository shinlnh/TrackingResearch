from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TrackerRow:
    tracker: str
    fps: float
    success_rate: float


@dataclass(frozen=True)
class TrackerScore:
    rank: int
    tracker: str
    fps: float
    success_rate: float
    success_norm: float
    fps_mean_target_score: float
    combined_score: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Compute a tracker score using mean FPS as the target in the "
            "formula 0.6 * norm(SuccessRate) + 0.4 * exp(-|FPS - meanFPS| / 1)."
        )
    )
    parser.add_argument(
        "--correlation-csv",
        type=Path,
        default=repo_root / "OtherTracker" / "fps_successrate_correlation.csv",
        help="CSV containing tracker, fps, success_rate columns.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "OtherTracker" / "fps_successrate_meanfps_score.csv",
        help="Output CSV with the ranked score.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=repo_root / "OtherTracker" / "fps_successrate_meanfps_score.png",
        help="Output ranking chart.",
    )
    parser.add_argument(
        "--success-weight",
        type=float,
        default=0.6,
        help="Weight assigned to normalized SuccessRate.",
    )
    parser.add_argument(
        "--fps-decay",
        type=float,
        default=1.0,
        help="Decay denominator in exp(-|FPS - meanFPS| / decay).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[TrackerRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            TrackerRow(
                tracker=(row.get("tracker") or "").strip(),
                fps=float(row["fps"]),
                success_rate=float(row["success_rate"]),
            )
            for row in reader
            if (row.get("tracker") or "").strip()
        ]
    if not rows:
        raise ValueError(f"No valid rows found in {path}.")
    return rows


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.ones_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def build_scores(
    rows: list[TrackerRow],
    success_weight: float,
    fps_decay: float,
) -> tuple[list[TrackerScore], float]:
    if not 0.0 <= success_weight <= 1.0:
        raise ValueError("--success-weight must be between 0 and 1.")
    if fps_decay <= 0.0:
        raise ValueError("--fps-decay must be positive.")

    fps = np.asarray([row.fps for row in rows], dtype=np.float64)
    success = np.asarray([row.success_rate for row in rows], dtype=np.float64)
    mean_fps = float(fps.mean())

    success_norm = minmax_normalize(success)
    fps_mean_target_score = np.exp(-np.abs(fps - mean_fps) / fps_decay)
    fps_weight = 1.0 - success_weight
    combined = success_weight * success_norm + fps_weight * fps_mean_target_score

    indexed = list(
        zip(
            rows,
            success_norm.tolist(),
            fps_mean_target_score.tolist(),
            combined.tolist(),
            strict=True,
        )
    )
    indexed.sort(key=lambda item: (-item[3], -item[0].success_rate, -item[0].fps, item[0].tracker))

    scores: list[TrackerScore] = []
    for rank, (row, success_score, fps_score, total_score) in enumerate(indexed, 1):
        scores.append(
            TrackerScore(
                rank=rank,
                tracker=row.tracker,
                fps=row.fps,
                success_rate=row.success_rate,
                success_norm=success_score,
                fps_mean_target_score=fps_score,
                combined_score=total_score,
            )
        )
    return scores, mean_fps


def write_csv(scores: list[TrackerScore], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "tracker",
                "fps",
                "success_rate",
                "success_norm",
                "fps_mean_target_score",
                "combined_score",
            ],
        )
        writer.writeheader()
        for row in scores:
            writer.writerow(
                {
                    "rank": row.rank,
                    "tracker": row.tracker,
                    "fps": f"{row.fps:.6f}",
                    "success_rate": f"{row.success_rate:.6f}",
                    "success_norm": f"{row.success_norm:.6f}",
                    "fps_mean_target_score": f"{row.fps_mean_target_score:.6f}",
                    "combined_score": f"{row.combined_score:.6f}",
                }
            )


def plot_chart(
    scores: list[TrackerScore],
    out_png: Path,
    mean_fps: float,
    success_weight: float,
    fps_decay: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trackers = [row.tracker for row in scores][::-1]
    values = [row.combined_score for row in scores][::-1]
    colors = ["#d55e00" if tracker == "MyTracker" else "#4c78a8" for tracker in trackers]

    fig, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    bars = ax.barh(trackers, values, color=colors, edgecolor="#1f1f1f", linewidth=0.7)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlabel("MeanFPS Target Score")
    ax.set_ylabel("Tracker")
    ax.set_title(
        "OTB100 Tracker Score Using meanFPS Target\n"
        f"Score = {success_weight:.2f} * norm(SuccessRate) + "
        f"{1.0 - success_weight:.2f} * exp(-|FPS - {mean_fps:.3f}| / {fps_decay:.1f})"
    )

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0.0, max(values) + 0.14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.correlation_csv)
    scores, mean_fps = build_scores(rows, args.success_weight, args.fps_decay)
    write_csv(scores, args.out_csv)
    plot_chart(scores, args.out_png, mean_fps, args.success_weight, args.fps_decay)

    print(f"Wrote score CSV: {args.out_csv}")
    print(f"Wrote score chart: {args.out_png}")
    print(f"meanFPS={mean_fps:.6f}")
    print(
        "ScoreFormula="
        f"{args.success_weight:.2f}*norm(SuccessRate)+"
        f"{1.0 - args.success_weight:.2f}*exp(-|FPS-{mean_fps:.6f}|/{args.fps_decay:.1f})"
    )
    winner = scores[0]
    print(
        f"Winner={winner.tracker} "
        f"(combined_score={winner.combined_score:.6f}, "
        f"success_rate={winner.success_rate:.6f}, fps={winner.fps:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
