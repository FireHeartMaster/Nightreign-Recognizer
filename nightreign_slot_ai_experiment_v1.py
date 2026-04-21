#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

LABELS = [
    "empty",
    "Church",
    "Fort",
    "Fortified Village",
    "Great Church",
    "Main Encampment",
    "Ruins",
    "Sorcerer's Rise",
    "Swamp",
    "Township",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}


@dataclass
class PatchSample:
    image_path: Path
    source_tag: str
    layout_number: int
    earth_name: str
    slot_id: int
    slot_name: str
    label_name: str
    patch_bgr: np.ndarray


class SlotPatchDataset(Dataset):
    def __init__(self, samples: Sequence[PatchSample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_rgb = cv2.cvtColor(s.patch_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            x = self.transform(img_rgb)
        else:
            x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        y = LABEL_TO_IDX[s.label_name]
        meta = {
            "image_path": str(s.image_path),
            "source_tag": s.source_tag,
            "layout_number": s.layout_number,
            "earth_name": s.earth_name,
            "slot_id": s.slot_id,
            "slot_name": s.slot_name,
            "label_name": s.label_name,
        }
        return x, y, meta


class SmallSlotCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Small AI experiment for Nightreign option B: train a slot-patch classifier on one or more "
            "reference-image folders using slot labels from the .npz package."
        )
    )
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package.")
    p.add_argument(
        "--images-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing numbered panel images. Repeat to include multiple sources (e.g. original and sim1080).",
    )
    p.add_argument("--patch-size", type=int, default=96, help="Final patch size fed to the model.")
    p.add_argument("--context-scale", type=float, default=1.8, help="Multiplier applied to slot radius when extracting a patch.")
    p.add_argument("--min-half-frac", type=float, default=0.075, help="Minimum half-size as a fraction of panel side.")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--balance-sampler", action="store_true", help="Use a weighted sampler to reduce class imbalance.")
    p.add_argument("--max-layouts-per-dir", type=int, default=0, help="Optional cap per source directory. 0 means all.")
    p.add_argument("--output-dir", type=Path, default=Path("slot_ai_experiment_output"))
    p.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
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


def list_numbered_images(images_dir: Path, max_layouts: int = 0) -> List[Tuple[int, Path]]:
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
    if max_layouts > 0:
        items = items[:max_layouts]
    return items


def load_package_metadata(package_path: Path) -> Dict[str, Any]:
    with np.load(package_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    return metadata


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
        img_bgr = cv2.copyMakeBorder(
            img_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    patch = img_bgr[y0:y1, x0:x1]
    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        raise ValueError("Empty patch extracted.")
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return patch


def build_samples(
    metadata: Dict[str, Any],
    image_dirs: Sequence[Path],
    patch_size: int,
    context_scale: float,
    min_half_frac: float,
    max_layouts_per_dir: int,
) -> List[PatchSample]:
    layouts = {int(item["layout_number"]): item for item in metadata["layouts"]}
    slots_by_earth: Dict[str, List[dict]] = {k: list(v) for k, v in metadata["slots_by_earth"].items()}

    samples: List[PatchSample] = []
    for images_dir in image_dirs:
        source_tag = images_dir.name
        numbered = list_numbered_images(images_dir, max_layouts_per_dir)
        print(f"Loading {len(numbered)} images from {images_dir} (source_tag={source_tag})")
        for layout_number, image_path in numbered:
            layout = layouts.get(layout_number)
            if layout is None:
                continue
            earth_name = str(layout["shifting_earth"])
            gt_slot_labels = list(layout["slot_labels"])
            slot_defs = slots_by_earth[earth_name]

            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"Warning: failed to read {image_path}")
                continue
            h, w = img_bgr.shape[:2]
            side = min(h, w)
            if h != w:
                # Use only the square top-left if needed, but these panels should already be square.
                img_bgr = img_bgr[:side, :side].copy()
                h = w = side

            if len(slot_defs) != len(gt_slot_labels):
                print(f"Warning: slot count mismatch for layout #{layout_number} in {image_path.name}")
                continue

            for slot_def, gt_label in zip(slot_defs, gt_slot_labels):
                cx = int(round(float(slot_def["x_norm"]) * w))
                cy = int(round(float(slot_def["y_norm"]) * h))
                base_half = int(round(max(float(slot_def["radius_norm"]) * side * float(context_scale), float(min_half_frac) * side)))
                half = max(18, base_half)
                patch = reflect_pad_extract(img_bgr, cx, cy, half, patch_size)
                samples.append(
                    PatchSample(
                        image_path=image_path,
                        source_tag=source_tag,
                        layout_number=layout_number,
                        earth_name=earth_name,
                        slot_id=int(slot_def["slot_id"]),
                        slot_name=str(slot_def["slot_name"]),
                        label_name=str(gt_label),
                        patch_bgr=patch,
                    )
                )
    if not samples:
        raise RuntimeError("No patch samples were built.")
    return samples


def split_layouts_by_group(samples: Sequence[PatchSample], train_ratio: float, val_ratio: float, seed: int):
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
        # fallback
        test_layouts = {unique_layouts[-1]}
        if unique_layouts[-1] in train_layouts:
            train_layouts.remove(unique_layouts[-1])

    train_samples = [s for s in samples if s.layout_number in train_layouts]
    val_samples = [s for s in samples if s.layout_number in val_layouts]
    test_samples = [s for s in samples if s.layout_number in test_layouts]

    return train_samples, val_samples, test_samples


def make_transforms(patch_size: int):
    train_tf = T.Compose([
        T.ToPILImage(),
        T.RandomApply([T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.02)], p=0.6),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.3, 1.1))], p=0.35),
        T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06)),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
    ])
    return train_tf, eval_tf


def build_sampler(samples: Sequence[PatchSample]) -> WeightedRandomSampler:
    counts = Counter(s.label_name for s in samples)
    weights = [1.0 / max(1, counts[s.label_name]) for s in samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def compute_class_weights(samples: Sequence[PatchSample]) -> torch.Tensor:
    counts = Counter(s.label_name for s in samples)
    total = sum(counts.values())
    weights = []
    for name in LABELS:
        c = max(1, counts.get(name, 0))
        weights.append(total / (len(LABELS) * c))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    by_class_total = Counter()
    by_class_correct = Counter()
    confusion = defaultdict(Counter)
    by_source_total = Counter()
    by_source_correct = Counter()
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y, meta in loader:
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

        # meta is dict of lists due default collate
        source_tags = meta["source_tag"]
        for i, (yt, yp) in enumerate(zip(y_cpu, pred_cpu)):
            y_name = LABELS[int(yt)]
            p_name = LABELS[int(yp)]
            by_class_total[y_name] += 1
            by_source_total[source_tags[i]] += 1
            if yt == yp:
                by_class_correct[y_name] += 1
                by_source_correct[source_tags[i]] += 1
            confusion[y_name][p_name] += 1

    metrics = {
        "loss": (loss_sum / max(1, total)),
        "acc": (correct / max(1, total)),
        "per_class_acc": {
            name: (by_class_correct[name] / max(1, by_class_total[name]))
            for name in LABELS
        },
        "per_source_acc": {
            name: (by_source_correct[name] / max(1, by_source_total[name]))
            for name in sorted(by_source_total.keys())
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "count": total,
    }
    return metrics


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

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
    }


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
    samples = build_samples(
        metadata=metadata,
        image_dirs=args.images_dir,
        patch_size=args.patch_size,
        context_scale=args.context_scale,
        min_half_frac=args.min_half_frac,
        max_layouts_per_dir=args.max_layouts_per_dir,
    )
    print(f"Built {len(samples)} patch samples.")

    label_counts = Counter(s.label_name for s in samples)
    print("Label counts:")
    for name in LABELS:
        print(f"  {name:>16}: {label_counts.get(name, 0)}")

    source_counts = Counter(s.source_tag for s in samples)
    print("Source counts:")
    for name, count in source_counts.items():
        print(f"  {name}: {count}")

    train_samples, val_samples, test_samples = split_layouts_by_group(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Test samples:  {len(test_samples)}")

    train_tf, eval_tf = make_transforms(args.patch_size)

    ds_train = SlotPatchDataset(train_samples, transform=train_tf)
    ds_val = SlotPatchDataset(val_samples, transform=eval_tf)
    ds_test = SlotPatchDataset(test_samples, transform=eval_tf)

    sampler = build_sampler(train_samples) if args.balance_sampler else None
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallSlotCNN(num_classes=len(LABELS)).to(device)
    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_metrics = evaluate_model(model, dl_val, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_per_source_acc": val_metrics["per_source_acc"],
        }
        history.append(row)
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

    final_val = evaluate_model(model, dl_val, device)
    final_test = evaluate_model(model, dl_test, device)

    model_path = output_dir / "slot_patch_model_v1.pt"
    labels_path = output_dir / "slot_patch_model_labels.json"
    report_path = output_dir / "slot_patch_model_report.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "patch_size": args.patch_size,
            "context_scale": args.context_scale,
            "min_half_frac": args.min_half_frac,
            "metadata": {
                "package": str(args.package),
                "images_dirs": [str(p) for p in args.images_dir],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        },
        model_path,
    )
    labels_path.write_text(json.dumps({"labels": LABELS}, indent=2), encoding="utf-8")

    report = {
        "device": str(device),
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
        "label_counts_total": dict(label_counts),
        "source_counts_total": dict(source_counts),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "history": history,
        "final_val": final_val,
        "final_test": final_test,
        "args": {
            "package": str(args.package),
            "images_dirs": [str(p) for p in args.images_dir],
            "patch_size": args.patch_size,
            "context_scale": args.context_scale,
            "min_half_frac": args.min_half_frac,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "balance_sampler": args.balance_sampler,
            "max_layouts_per_dir": args.max_layouts_per_dir,
            "output_dir": str(args.output_dir),
        },
    }
    save_json(report_path, report)

    print("\n========== FINAL AI SLOT TEST ==========")
    print(f"Model path:       {model_path}")
    print(f"Model size (MB):  {report['model_size_mb']:.3f}")
    print(f"Final val acc:    {final_val['acc']:.4f}")
    print(f"Final test acc:   {final_test['acc']:.4f}")
    print("Final test per-source accuracy:")
    for source_name, acc in final_test["per_source_acc"].items():
        print(f"  {source_name}: {acc:.4f}")
    print("Final test per-class accuracy:")
    for label_name in LABELS:
        print(f"  {label_name:>16}: {final_test['per_class_acc'].get(label_name, 0.0):.4f}")
    print(f"\nSaved report to:  {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
