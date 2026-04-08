import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_ROOT / "runs" / "yolo"
DEFAULT_EXPORT_PATH = SCRIPT_ROOT / "weights" / "yolo_person_only.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO into a person-only detector for MyPersonTracker.")
    parser.add_argument("--data", required=True, help="Dataset YAML path for Ultralytics training.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_DIR), help="Ultralytics project directory.")
    parser.add_argument("--name", default="person_only", help="Ultralytics run name.")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="Class ids to keep during training. Default keeps only class 0.")
    parser.add_argument("--single-cls", action="store_true", default=True, help="Train as a single-class detector.")
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Allow reusing the same run directory.")
    parser.add_argument("--cache", action="store_true", default=False, help="Enable Ultralytics dataset caching.")
    parser.add_argument("--export-best", default=str(DEFAULT_EXPORT_PATH), help="Copy best.pt here after training.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        classes=args.classes if args.classes else None,
        single_cls=args.single_cls,
        cache=args.cache,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(model.trainer.save_dir)
    best_path = Path(getattr(model.trainer, "best", save_dir / "weights" / "best.pt"))
    if not best_path.is_file():
        raise FileNotFoundError(f"Could not find trained best.pt at {best_path}")

    export_path = Path(args.export_best)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, export_path)

    print(f"best_checkpoint={best_path}")
    print(f"exported_checkpoint={export_path}")


if __name__ == "__main__":
    main()
