from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SPECIAL_CASES = {
    "Board": {"nz": 5},
    "David": {"start_frame": 300, "end_frame": 770},
    "Football1": {"end_frame": 74},
    "Freeman3": {"end_frame": 460},
    "Freeman4": {"end_frame": 283},
    "BlurCar1": {"start_frame": 247, "end_frame": 988},
    "BlurCar3": {"start_frame": 3, "end_frame": 359},
    "BlurCar4": {"start_frame": 18, "end_frame": 397},
    "Tiger1": {"start_frame": 6, "end_frame": 354},
}


@dataclass(frozen=True)
class OTBSequence:
    name: str
    base_name: str
    split_id: str | None
    path: Path
    anno_file: Path
    image_dir: Path
    nz: int
    ext: str
    start_frame: int
    end_frame: int
    groundtruth_rects: tuple[tuple[float, float, float, float], ...]
    init_rect: tuple[float, float, float, float]

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def frame_path(self, frame_index: int) -> Path:
        return self.image_dir / f"{frame_index:0{self.nz}d}.{self.ext}"

    def frame_paths(self) -> list[Path]:
        return [self.frame_path(i) for i in range(self.start_frame, self.end_frame + 1)]


def _read_rects(path: Path) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
        vals = [float(p) for p in parts[:4]]
        rects.append((vals[0], vals[1], vals[2], vals[3]))
    if not rects:
        raise ValueError(f"No annotations found in {path}")
    return rects


def load_otb_sequences(otb_root: Path, sequence_file: Path) -> list[OTBSequence]:
    sequence_names = [
        line.strip()
        for line in sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    sequences: list[OTBSequence] = []
    for sequence_name in sequence_names:
        split_parts = sequence_name.split("-", maxsplit=1)
        base_name = split_parts[0]
        split_id = split_parts[1] if len(split_parts) == 2 else None

        seq_path = otb_root / base_name
        anno_name = f"groundtruth_rect.{split_id}.txt" if split_id else "groundtruth_rect.txt"
        anno_file = seq_path / anno_name
        rects = _read_rects(anno_file)

        nz = 4
        ext = "jpg"
        start_frame = 1
        end_frame = len(rects)

        case = SPECIAL_CASES.get(sequence_name, {})
        nz = case.get("nz", nz)
        start_frame = case.get("start_frame", start_frame)
        end_frame = case.get("end_frame", end_frame)

        expected_length = end_frame - start_frame + 1
        if len(rects) != expected_length:
            rects = rects[start_frame - 1 : end_frame]

        if len(rects) != expected_length:
            raise ValueError(
                f"Sequence {sequence_name} expected {expected_length} annotations, found {len(rects)}"
            )

        init_rect = rects[0]

        sequences.append(
            OTBSequence(
                name=sequence_name,
                base_name=base_name,
                split_id=split_id,
                path=seq_path,
                anno_file=anno_file,
                image_dir=seq_path / "img",
                nz=nz,
                ext=ext,
                start_frame=start_frame,
                end_frame=end_frame,
                groundtruth_rects=tuple(rects),
                init_rect=init_rect,
            )
        )

    return sequences
