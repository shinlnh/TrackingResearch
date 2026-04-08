import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a balanced CASIA-WebFace subset for face embedding fine-tuning.")
    parser.add_argument("--dataset", default="SaffalPoosh/casia_web_face", help="Hugging Face dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--output", default="MyPersonTracker/datasets/casia_webface_balanced", help="Output directory.")
    parser.add_argument("--max-identities", type=int, default=256, help="Number of identities to keep.")
    parser.add_argument("--train-per-identity", type=int, default=20, help="Training images per identity.")
    parser.add_argument("--val-per-identity", type=int, default=4, help="Validation images per identity.")
    return parser


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_subset(dataset_name: str, split: str, output_dir: Path, max_identities: int, train_per_identity: int, val_per_identity: int) -> None:
    required_per_identity = train_per_identity + val_per_identity
    dataset = load_dataset(dataset_name, split=split)
    label_counts = Counter(dataset["label"])

    selected_identities = [
        label
        for label, count in label_counts.most_common()
        if count >= required_per_identity
    ][:max_identities]
    if not selected_identities:
        raise RuntimeError("No identities satisfy the requested train/val quota.")

    selected_set = set(selected_identities)
    samples_by_label: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(dataset["label"]):
        if label in selected_set and len(samples_by_label[label]) < required_per_identity:
            samples_by_label[label].append(index)
        if len(samples_by_label) == len(selected_identities) and all(
            len(indices) >= required_per_identity for indices in samples_by_label.values()
        ):
            break

    train_root = output_dir / "train"
    val_root = output_dir / "val"
    ensure_clean_dir(train_root)
    ensure_clean_dir(val_root)

    manifest = []
    for new_label, original_label in enumerate(selected_identities):
        indices = samples_by_label[original_label]
        if len(indices) < required_per_identity:
            continue

        train_indices = indices[:train_per_identity]
        val_indices = indices[train_per_identity:required_per_identity]
        train_class_dir = train_root / f"{new_label:04d}"
        val_class_dir = val_root / f"{new_label:04d}"
        ensure_clean_dir(train_class_dir)
        ensure_clean_dir(val_class_dir)

        for local_idx, dataset_idx in enumerate(train_indices):
            image = dataset[dataset_idx]["image"].convert("RGB")
            image.save(train_class_dir / f"{local_idx:03d}.png")

        for local_idx, dataset_idx in enumerate(val_indices):
            image = dataset[dataset_idx]["image"].convert("RGB")
            image.save(val_class_dir / f"{local_idx:03d}.png")

        manifest.append(
            {
                "new_label": new_label,
                "original_label": int(original_label),
                "count": len(indices),
                "train_images": len(train_indices),
                "val_images": len(val_indices),
            }
        )

    metadata = {
        "dataset": dataset_name,
        "split": split,
        "max_identities": max_identities,
        "train_per_identity": train_per_identity,
        "val_per_identity": val_per_identity,
        "selected_identities": len(manifest),
    }
    (output_dir / "manifest.json").write_text(json.dumps({"metadata": metadata, "classes": manifest}, indent=2), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    save_subset(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=Path(args.output),
        max_identities=args.max_identities,
        train_per_identity=args.train_per_identity,
        val_per_identity=args.val_per_identity,
    )


if __name__ == "__main__":
    main()
