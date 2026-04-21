#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class EarthPanelCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prepare_panel_for_earth_ai(panel_bgr: np.ndarray, panel_size: int) -> np.ndarray:
    side = min(panel_bgr.shape[:2])
    if panel_bgr.shape[0] != panel_bgr.shape[1]:
        y0 = max(0, (panel_bgr.shape[0] - side) // 2)
        x0 = max(0, (panel_bgr.shape[1] - side) // 2)
        panel_bgr = panel_bgr[y0:y0 + side, x0:x0 + side].copy()
    else:
        panel_bgr = panel_bgr.copy()

    side = panel_bgr.shape[0]
    cy = (side - 1) / 2.0
    cx = (side - 1) / 2.0
    yy, xx = np.ogrid[:side, :side]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer_radius = side * 0.47
    outside_mask = dist >= outer_radius

    neutral = np.full_like(panel_bgr, 168)
    panel_bgr[outside_mask] = neutral[outside_mask]

    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 70, 60], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
    gold_mask = cv2.inRange(hsv, np.array([8, 80, 90], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 210], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
    overlay_mask = cv2.bitwise_or(blue_mask, gold_mask)
    overlay_mask = cv2.bitwise_or(overlay_mask, white_mask)
    overlay_mask[outside_mask] = 0
    overlay_mask = cv2.dilate(overlay_mask, np.ones((5, 5), np.uint8), iterations=1)
    if int(np.count_nonzero(overlay_mask)) > 0:
        panel_bgr = cv2.inpaint(panel_bgr, overlay_mask, 3, cv2.INPAINT_TELEA)

    if panel_bgr.shape[0] != panel_size or panel_bgr.shape[1] != panel_size:
        panel_bgr = cv2.resize(panel_bgr, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    return panel_bgr


@dataclass
class EarthSample:
    image_path: Path
    layout_number: int
    earth_name: str
    panel_bgr: np.ndarray


class EarthPanelDataset(Dataset):
    def __init__(self, samples: Sequence[EarthSample], earth_names: Sequence[str], transform=None):
        self.samples = list(samples)
        self.earth_names = list(earth_names)
        self.earth_to_idx = {name: idx for idx, name in enumerate(self.earth_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_rgb = cv2.cvtColor(s.panel_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb) if self.transform is not None else torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        y = self.earth_to_idx[s.earth_name]
        meta = {
            "image_path": str(s.image_path),
            "layout_number": s.layout_number,
            "earth_name": s.earth_name,
        }
        return x, y, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a panel-level Shifting Earth classifier from rendered layout images.")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, action="append", required=True)
    p.add_argument("--panel-size", type=int, default=256)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("earth_ai_test_output"))
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_package_metadata(package_path: Path) -> Dict[str, Any]:
    with np.load(package_path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"]))


def list_numbered_images(images_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        items.append((int(m.group(1)), path))
    if not items:
        raise FileNotFoundError(f"No numbered images found in {images_dir}")
    return items


def resize_panel(img_bgr: np.ndarray, panel_size: int) -> np.ndarray:
    if img_bgr.shape[0] != panel_size or img_bgr.shape[1] != panel_size:
        return cv2.resize(img_bgr, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    return img_bgr


def build_samples(metadata: Dict[str, Any], image_dirs: Sequence[Path], panel_size: int) -> Tuple[List[EarthSample], List[str]]:
    layouts = {int(item["layout_number"]): item for item in metadata["layouts"]}
    earth_names = sorted({str(item["shifting_earth"]) for item in metadata["layouts"]})
    samples: List[EarthSample] = []
    for images_dir in image_dirs:
        numbered = list_numbered_images(images_dir)
        print(f"Loading {len(numbered)} images from {images_dir}")
        for layout_number, image_path in numbered:
            layout = layouts.get(layout_number)
            if layout is None:
                continue
            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_bgr = prepare_panel_for_earth_ai(img_bgr, panel_size)
            samples.append(EarthSample(image_path=image_path, layout_number=layout_number, earth_name=str(layout["shifting_earth"]), panel_bgr=img_bgr))
    if not samples:
        raise RuntimeError("No earth-panel samples were built.")
    return samples, earth_names


def split_layouts_by_group(samples: Sequence[EarthSample], train_ratio: float, val_ratio: float, seed: int):
    unique_layouts = sorted({int(s.layout_number) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(unique_layouts)
    n = len(unique_layouts)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, min(n_val, n - 2))
        n_train = max(1, n - n_val - 1)
    train_layouts = set(unique_layouts[:n_train])
    val_layouts = set(unique_layouts[n_train:n_train + n_val])
    test_layouts = set(unique_layouts[n_train + n_val:])
    if not test_layouts:
        test_layouts = {unique_layouts[-1]}
        train_layouts.discard(unique_layouts[-1])
    train_samples = [s for s in samples if s.layout_number in train_layouts]
    val_samples = [s for s in samples if s.layout_number in val_layouts]
    test_samples = [s for s in samples if s.layout_number in test_layouts]
    return train_samples, val_samples, test_samples


def make_transforms():
    train_tf = T.Compose([
        T.ToPILImage(),
        T.RandomApply([T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)], p=0.6),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.3, 1.2))], p=0.25),
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([T.ToTensor()])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, earth_names: Sequence[str], device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    by_class_total = Counter()
    by_class_correct = Counter()
    confusion: Dict[str, Counter] = {name: Counter() for name in earth_names}
    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        loss_sum += float(loss.item()) * int(x.shape[0])
        total += int(x.shape[0])
        correct += int((pred == y).sum().item())
        pred_cpu = pred.cpu().numpy().tolist()
        y_cpu = y.cpu().numpy().tolist()
        for yt, yp in zip(y_cpu, pred_cpu):
            y_name = earth_names[int(yt)]
            p_name = earth_names[int(yp)]
            by_class_total[y_name] += 1
            if yt == yp:
                by_class_correct[y_name] += 1
            confusion[y_name][p_name] += 1
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "per_class_acc": {name: by_class_correct[name] / max(1, by_class_total[name]) for name in earth_names},
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "count": total,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(logits, dim=1)
        total += int(x.shape[0])
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * int(x.shape[0])
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_package_metadata(args.package)
    samples, earth_names = build_samples(metadata, args.images_dir, args.panel_size)
    print(f"Built {len(samples)} earth-panel samples.")
    counts = Counter(s.earth_name for s in samples)
    for name in earth_names:
        print(f"{name:>14}: {counts.get(name, 0)}")
    train_samples, val_samples, test_samples = split_layouts_by_group(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Test samples:  {len(test_samples)}")
    train_tf, eval_tf = make_transforms()
    ds_train = EarthPanelDataset(train_samples, earth_names, transform=train_tf)
    ds_val = EarthPanelDataset(val_samples, earth_names, transform=eval_tf)
    ds_test = EarthPanelDataset(test_samples, earth_names, transform=eval_tf)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = EarthPanelCNN(num_classes=len(earth_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = -1.0
    best_state = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_metrics = evaluate_model(model, dl_val, earth_names, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        })
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_model(model, dl_val, earth_names, device)
    final_test = evaluate_model(model, dl_test, earth_names, device)
    model_path = output_dir / "earth_panel_model_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "earth_names": list(earth_names),
            "panel_size": int(args.panel_size),
        },
        model_path,
    )
    report = {
        "device": str(device),
        "model_path": str(model_path),
        "label_counts_total": dict(counts),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "history": history,
        "final_val": final_val,
        "final_test": final_test,
        "args": {
            "package": str(args.package),
            "images_dirs": [str(p) for p in args.images_dir],
            "panel_size": int(args.panel_size),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "num_workers": int(args.num_workers),
            "output_dir": str(args.output_dir),
        },
    }
    save_json(output_dir / "earth_panel_model_report.json", report)
    print("\n========== FINAL EARTH AI TEST ==========")
    print(f"Model path:       {model_path}")
    print(f"Final val acc:    {final_val['acc']:.4f}")
    print(f"Final test acc:   {final_test['acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
