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
LABELS = ["Status_Fire", "Status_Magic"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

GREAT_HOLLOW_PATTERNSET_DIRS = [Path("great_hollow_pattern_sets")]

PALACE_ANCHOR = (277.396, 805.636)
FORTRESS_ANCHOR = (737.111, 264.556)
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0
IMAGE_W = 1536.0
IMAGE_H = 1536.0


@dataclass
class StatusSample:
    image_path: Path
    layout_number: int
    earth_name: str
    anchor_name: str
    label_name: str
    patch_bgr: np.ndarray


class StatusDataset(Dataset):
    def __init__(self, samples: Sequence[StatusSample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_rgb = cv2.cvtColor(s.patch_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb) if self.transform is not None else torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        y = LABEL_TO_IDX[s.label_name]
        return x, y, {
            "image_path": str(s.image_path),
            "layout_number": s.layout_number,
            "earth_name": s.earth_name,
            "anchor_name": s.anchor_name,
            "label_name": s.label_name,
        }


class StatusCNN(nn.Module):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Great Hollow status-only classifier for Fire vs Magic.")
    p.add_argument("--images-dir", type=Path, action="append", required=True)
    p.add_argument("--patternset-dir", type=Path, default=Path("great_hollow_pattern_sets"))
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--context-scale", type=float, default=0.75)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("great_hollow_status_ai_test_output"))
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_patternset_labels(patternset_dir: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for path in sorted(patternset_dir.glob("pattern_*_mappatternset_bundle.json")):
        m = re.search(r"pattern_(\d+)_", path.name)
        if not m:
            continue
        layout_number = int(m.group(1))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        active = {}
        for item in payload.get("activeMapPatternSetFields", []) or []:
            field_name = str(item.get("field", "")).strip()
            try:
                value = int(item.get("value", 0))
            except (TypeError, ValueError):
                value = 0
            active[field_name] = value
        value = active.get("greatHollowWestFortMainModifier1")
        if value == 15100:
            label = "Status_Fire"
        elif value == 15101:
            label = "Status_Magic"
        else:
            value2 = active.get("greatHollowWestFortMainModifier2")
            if value2 == 15100:
                label = "Status_Fire"
            elif value2 == 15101:
                label = "Status_Magic"
            else:
                label = None
        if label in LABELS:
            out[layout_number] = label
    return out


def list_numbered_images(images_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        items.append((int(m.group(1)), path))
    return items


def world_to_panel_norm(x_world: float, z_world: float) -> Tuple[float, float]:
    px = (float(x_world) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)
    py = 1.0 - ((float(z_world) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z))
    return float(px), float(py)


def reflect_pad_extract(img_bgr: np.ndarray, cx: int, cy: int, half: int, patch_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0 = cx - half
    y0 = cy - half
    x1 = cx + half
    y1 = cy + half
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        img_bgr = cv2.copyMakeBorder(img_bgr, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101)
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    patch = img_bgr[y0:y1, x0:x1]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return patch


def build_samples(image_dirs: Sequence[Path], patternset_dir: Path, patch_size: int, context_scale: float) -> List[StatusSample]:
    layout_to_label = load_patternset_labels(patternset_dir)
    samples: List[StatusSample] = []
    for images_dir in image_dirs:
        for layout_number, image_path in list_numbered_images(images_dir):
            if layout_number not in layout_to_label:
                continue
            label = layout_to_label[layout_number]
            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            side = min(h, w)
            if h != w:
                img_bgr = img_bgr[:side, :side].copy()
                h = w = side
            anchor_name, anchor = ("Palace", PALACE_ANCHOR)
            x_norm, y_norm = world_to_panel_norm(*anchor)
            cx = int(round(x_norm * w))
            cy = int(round(y_norm * h))
            half = max(18, int(round(side * context_scale * 0.018)))
            patch = reflect_pad_extract(img_bgr, cx, cy, half, patch_size)
            samples.append(StatusSample(
                image_path=image_path,
                layout_number=layout_number,
                earth_name="Great Hollow",
                anchor_name=anchor_name,
                label_name=label,
                patch_bgr=patch,
            ))
    return samples


def split_layouts(samples: Sequence[StatusSample], train_ratio: float, val_ratio: float, seed: int):
    layouts = sorted({s.layout_number for s in samples})
    rng = random.Random(seed)
    rng.shuffle(layouts)
    n = len(layouts)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, min(n_val, n - 2))
        n_train = max(1, n - n_val - 1)
    train_layouts = set(layouts[:n_train])
    val_layouts = set(layouts[n_train:n_train + n_val])
    test_layouts = set(layouts[n_train + n_val:])
    if not test_layouts and val_layouts:
        test_layouts.add(val_layouts.pop())
    train = [s for s in samples if s.layout_number in train_layouts]
    val = [s for s in samples if s.layout_number in val_layouts]
    test = [s for s in samples if s.layout_number in test_layouts]
    return train, val, test


def make_transforms(patch_size: int):
    train_tf = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=0.08, contrast=0.10, saturation=0.12, hue=0.02),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])
    return train_tf, eval_tf


def accuracy(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            pred = logits.argmax(dim=1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            loss_sum += float(loss.item()) * int(y.numel())
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    samples = build_samples(args.images_dir, args.patternset_dir, args.patch_size, args.context_scale)
    if not samples:
        raise RuntimeError("No samples were built.")
    train_samples, val_samples, test_samples = split_layouts(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train labels: {Counter(s.label_name for s in train_samples)}")
    train_tf, eval_tf = make_transforms(args.patch_size)
    ds_train = StatusDataset(train_samples, transform=train_tf)
    ds_val = StatusDataset(val_samples, transform=eval_tf)
    ds_test = StatusDataset(test_samples, transform=eval_tf)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = StatusCNN(num_classes=len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        correct = 0
        for x, y, _meta in dl_train:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_n += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())
        train_loss = total_loss / max(1, total_n)
        train_acc = correct / max(1, total_n)
        val_metrics = accuracy(model, dl_val, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"]})
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = accuracy(model, dl_test, device)
    print(f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_model = args.output_dir / "great_hollow_status_model_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": LABELS,
        "patch_size": int(args.patch_size),
        "context_scale": float(args.context_scale),
        "min_half_frac": 0.018,
        "history": history,
        "test_metrics": test_metrics,
    }, out_model)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "test_metrics": test_metrics,
        "best_val_acc": best_val,
    }, indent=2), encoding="utf-8")
    print(f"Saved model: {out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
