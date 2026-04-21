import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import layout_classifier_lib as lib


class Slot6Dataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        image_size: int = 200,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            shift_x = random.randint(-6, 6)
            shift_y = random.randint(-6, 6)
            if shift_x or shift_y:
                mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                img = cv2.warpAffine(img, mat, (self.image_size, self.image_size), borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.5:
                alpha = random.uniform(0.9, 1.1)
                beta = random.uniform(-12, 12)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        x = lib.preprocess_bgr_200(img, image_size=self.image_size)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def split_train_val_per_class(
    samples: Sequence[Tuple[Path, int]],
    num_classes: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    by_class: Dict[int, List[Tuple[Path, int]]] = {i: [] for i in range(num_classes)}
    for item in samples:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    for _, items in by_class.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, max(1, len(shuffled) - 1))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    sum_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            sum_loss += float(loss.item()) * int(y.numel())
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": sum_loss / total, "accuracy": correct / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Slot 6 specific multi-class CNN.")
    parser.add_argument("--data-dir", default=".", help="Folder containing class subfolders.")
    parser.add_argument("--output-model", default="slot6_status_cnn.pt", help="Output model file.")
    parser.add_argument("--output-report", default="training_report.json", help="Output training report JSON.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    class_names = lib.discover_class_names(data_dir)
    if len(class_names) < 2:
        raise SystemExit(f"Expected at least 2 class folders, found: {class_names}")

    samples = lib.collect_samples(data_dir, class_names)
    if not samples:
        raise SystemExit("No training images found.")

    train_samples, val_samples = split_train_val_per_class(
        samples=samples,
        num_classes=len(class_names),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    train_ds = Slot6Dataset(train_samples, image_size=lib.IMAGE_SIZE, augment=True)
    val_ds = Slot6Dataset(val_samples, image_size=lib.IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lib.Slot6StatusCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    out_model_path = Path(args.output_model).resolve()
    out_report_path = Path(args.output_report).resolve()

    print(f"Data dir: {data_dir}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
    print(f"Device: {device}")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            n = int(y.numel())
            running_total += n
            running_correct += int((pred == y).sum().item())
            running_loss += float(loss.item()) * n

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{int(args.epochs):02d} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        )

        if row["val_accuracy"] > best_val_acc:
            best_val_acc = row["val_accuracy"]
            best_epoch = epoch
            meta = {
                "best_epoch": int(best_epoch),
                "best_val_accuracy": float(best_val_acc),
                "train_samples": int(len(train_samples)),
                "val_samples": int(len(val_samples)),
            }
            lib.save_checkpoint(out_model_path, model, class_names, lib.IMAGE_SIZE, meta)

    report = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "model_path": str(out_model_path),
        "history": history,
    }
    lib.write_json(out_report_path, report)
    print(f"\nBest val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {out_model_path}")
    print(f"Report saved to: {out_report_path}")


if __name__ == "__main__":
    main()
