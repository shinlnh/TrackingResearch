import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune InceptionResnetV1 on an identity-labeled face dataset.")
    parser.add_argument("--data-root", default="MyPersonTracker/datasets/casia_webface_balanced", help="Dataset root containing train/ and val/.")
    parser.add_argument("--pretrained", default="casia-webface", choices=["casia-webface", "vggface2"], help="Starting checkpoint for InceptionResnetV1.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--output", default="MyPersonTracker/weights/face_embedding_casia_balanced.pt", help="Output checkpoint path.")
    parser.add_argument("--project", default="MyPersonTracker/runs/face_embedding", help="Directory for logs and best checkpoint.")
    parser.add_argument("--name", default="casia_balanced_ft", help="Run name.")
    return parser


def build_transforms(train: bool) -> transforms.Compose:
    augments = [transforms.Resize((160, 160))]
    if train:
        augments.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            ]
        )
    augments.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(fixed_image_standardization),
        ]
    )
    return transforms.Compose(augments)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise SystemExit(f"Expected train/ and val/ under {data_root}")

    train_dataset = datasets.ImageFolder(train_root, transform=build_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_root, transform=build_transforms(train=False))
    num_classes = len(train_dataset.classes)
    if num_classes == 0:
        raise SystemExit("No classes found in training dataset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = InceptionResnetV1(pretrained=args.pretrained, classify=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    history = []

    def build_checkpoint() -> Dict[str, object]:
        state_dict = model.state_dict()
        embedding_state_dict = {key: value for key, value in state_dict.items() if not key.startswith("logits.")}
        return {
            "state_dict": state_dict,
            "embedding_state_dict": embedding_state_dict,
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "class_to_idx": train_dataset.class_to_idx,
            "history": history,
        }

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_samples += labels.size(0)

        train_loss = train_loss_sum / max(1, train_samples)
        train_acc = train_correct / max(1, train_samples)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_record: Dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_record)
        print(epoch_record)

        checkpoint = build_checkpoint()
        torch.save(checkpoint, run_dir / "last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, run_dir / "best.pt")
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
