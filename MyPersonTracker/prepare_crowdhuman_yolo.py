import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CrowdHuman annotations to YOLO person-only detection format.")
    parser.add_argument("--raw-root", default="MyPersonTracker/datasets/crowdhuman_raw", help="Directory containing CrowdHuman zips and odgt files.")
    parser.add_argument("--images-root", default="MyPersonTracker/datasets/crowdhuman_extracted/Images", help="Directory containing extracted CrowdHuman images.")
    parser.add_argument("--output-root", default="MyPersonTracker/datasets/crowdhuman_yolo", help="YOLO dataset output directory.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="How to materialize images into YOLO split folders.")
    parser.add_argument("--box-key", choices=["fbox", "vbox"], default="fbox", help="Bounding box key to use.")
    return parser


def safe_link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def convert_box_to_yolo(x: float, y: float, w: float, h: float, image_w: int, image_h: int) -> str | None:
    if w <= 1 or h <= 1:
        return None

    x_center = (x + w / 2.0) / image_w
    y_center = (y + h / 2.0) / image_h
    width = w / image_w
    height = h / image_h

    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return None

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_split(raw_root: Path, images_root: Path, output_root: Path, split: str, box_key: str, link_mode: str) -> None:
    odgt_path = raw_root / f"annotation_{split}.odgt"
    images_out = output_root / "images" / split
    labels_out = output_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with odgt_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc=f"convert-{split}"):
        record = json.loads(line)
        image_id = record["ID"]
        src_image = images_root / f"{image_id}.jpg"
        if not src_image.is_file():
            raise FileNotFoundError(f"Missing image for {image_id}: {src_image}")

        dst_image = images_out / f"{image_id}.jpg"
        safe_link_or_copy(src_image, dst_image, link_mode)

        with Image.open(src_image) as image:
            image_w, image_h = image.size

        label_lines: list[str] = []
        for gtbox in record.get("gtboxes", []):
            if gtbox.get("tag") != "person":
                continue
            extra = gtbox.get("extra", {})
            if int(extra.get("ignore", 0)) == 1:
                continue
            box = gtbox.get(box_key)
            if not box or len(box) != 4:
                continue
            label = convert_box_to_yolo(float(box[0]), float(box[1]), float(box[2]), float(box[3]), image_w, image_h)
            if label is not None:
                label_lines.append(label)

        label_path = labels_out / f"{image_id}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")


def write_dataset_yaml(output_root: Path) -> None:
    yaml_path = output_root / "crowdhuman_person.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: person",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_root = Path(args.raw_root)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)

    convert_split(raw_root, images_root, output_root, "train", args.box_key, args.link_mode)
    convert_split(raw_root, images_root, output_root, "val", args.box_key, args.link_mode)
    write_dataset_yaml(output_root)
    print(f"dataset_yaml={output_root / 'crowdhuman_person.yaml'}")


if __name__ == "__main__":
    main()
