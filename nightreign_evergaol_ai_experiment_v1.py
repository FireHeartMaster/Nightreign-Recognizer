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
IMAGE_W = 1536.0
IMAGE_H = 1536.0
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0
EVERGAOL_WORLD_POINTS: Tuple[Tuple[float, float], ...] = (
    (2.6, 793.03),
    (276.65, 309.65),
    (428.37, 693.18),
    (483.79, 978.09),
    (697.3, 101.45),
    (713.22, 722.15),
    (957.87, 530.38),
)
EVERGAOL_EXPECTED_BY_EARTH = {
    "Limveld": {0, 1, 2, 3, 4, 5, 6},
    "Mountaintop": {1, 4, 5, 6},
    "The Crater": {0, 1, 4, 6},
    "Rotted Woods": {0, 1, 2, 3, 5, 6},
    "Noklateo": {0, 2, 3, 4, 5, 6},
    "Great Hollow": set(),
}
LABELS = ["absent", "present"]


class EvergaolPatchCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
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


@dataclass
class EvergaolSample:
    image_path: Path
    layout_number: int
    earth_name: str
    evergaol_id: int
    label_name: str
    patch_bgr: np.ndarray


class EvergaolDataset(Dataset):
    def __init__(self, samples: Sequence[EvergaolSample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_rgb = cv2.cvtColor(s.patch_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb) if self.transform is not None else torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        y = 1 if s.label_name == "present" else 0
        meta = {
            "image_path": str(s.image_path),
            "layout_number": int(s.layout_number),
            "earth_name": str(s.earth_name),
            "evergaol_id": int(s.evergaol_id),
            "label_name": str(s.label_name),
        }
        return x, y, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a binary Evergaol patch detector from rendered layout images.")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, action="append", required=True)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--context-scale", type=float, default=2.0)
    p.add_argument("--min-half-frac", type=float, default=0.070)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("evergaol_ai_test_output"))
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
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return patch


def build_samples(metadata: Dict[str, Any], image_dirs: Sequence[Path], patch_size: int, context_scale: float, min_half_frac: float) -> List[EvergaolSample]:
    layouts = {int(item["layout_number"]): item for item in metadata["layouts"]}
    samples: List[EvergaolSample] = []
    for images_dir in image_dirs:
        numbered = list_numbered_images(images_dir)
        print(f"Loading {len(numbered)} images from {images_dir}")
        for layout_number, image_path in numbered:
            layout = layouts.get(layout_number)
            if layout is None:
                continue
            earth_name = str(layout["shifting_earth"])
            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            side = min(h, w)
            expected_present = set(EVERGAOL_EXPECTED_BY_EARTH.get(earth_name, set()))
            for idx, (x_world, z_world) in enumerate(EVERGAOL_WORLD_POINTS):
                x_norm, y_norm = world_to_panel_norm(x_world, z_world)
                cx = int(round(x_norm * w))
                cy = int(round(y_norm * h))
                half = int(round(max(side * context_scale * 0.018, min_half_frac * side)))
                half = max(20, half)
                patch = reflect_pad_extract(img_bgr, cx, cy, half, patch_size)
                samples.append(EvergaolSample(
                    image_path=image_path,
                    layout_number=int(layout_number),
                    earth_name=earth_name,
                    evergaol_id=int(idx + 1),
                    label_name="present" if idx in expected_present else "absent",
                    patch_bgr=patch,
                ))
    if not samples:
        raise RuntimeError("No Evergaol samples were built.")
    return samples


def split_layouts_by_group(samples: Sequence[EvergaolSample], train_ratio: float, val_ratio: float, seed: int):
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
    if not test_layouts and val_layouts:
        test_layouts.add(val_layouts.pop())
    train = [s for s in samples if s.layout_number in train_layouts]
    val = [s for s in samples if s.layout_number in val_layouts]
    test = [s for s in samples if s.layout_number in test_layouts]
    return train, val, test


def make_transforms(patch_size: int):
    train_tf = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=6),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])
    return train_tf, eval_tf


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    conf = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            for yi, pi in zip(y.cpu().numpy(), pred.cpu().numpy()):
                conf[int(yi), int(pi)] += 1
    acc = float(correct / max(1, total))
    return {"accuracy": acc, "confusion": conf.tolist()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_package_metadata(args.package)
    samples = build_samples(metadata, args.images_dir, args.patch_size, args.context_scale, args.min_half_frac)
    train_samples, val_samples, test_samples = split_layouts_by_group(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train labels: {Counter(s.label_name for s in train_samples)}")

    train_tf, eval_tf = make_transforms(args.patch_size)
    ds_train = EvergaolDataset(train_samples, transform=train_tf)
    ds_val = EvergaolDataset(val_samples, transform=eval_tf)
    ds_test = EvergaolDataset(test_samples, transform=eval_tf)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EvergaolPatchCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for x, y, _meta in dl_train:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_n += int(y.numel())
        train_loss = total_loss / max(1, total_n)
        val_metrics = evaluate_model(model, dl_val, device)
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_metrics["accuracy"]})
        if val_metrics["accuracy"] > best_val:
            best_val = float(val_metrics["accuracy"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no best_state.")
    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, dl_test, device)
    print(f"test_acc={test_metrics['accuracy']:.4f}")

    model_path = args.output_dir / "evergaol_patch_model_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "patch_size": int(args.patch_size),
            "context_scale": float(args.context_scale),
            "min_half_frac": float(args.min_half_frac),
        },
        model_path,
    )
    report = {
        "history": history,
        "best_val_accuracy": best_val,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_confusion": test_metrics["confusion"],
        "labels": LABELS,
        "patch_size": int(args.patch_size),
        "context_scale": float(args.context_scale),
        "min_half_frac": float(args.min_half_frac),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
