#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


POSITIVE_LABELS = {"Church", "Sorcerer's Rise"}
NEGATIVE_LABELS = {"empty"}


@dataclass
class Sample:
    image_path: Path
    layout_number: int
    slot_id: int
    slot_name: str
    gt_label: str
    binary_label: int  # 0=empty, 1=church_like
    patch_bgr: np.ndarray


class PatchDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], augment: bool = False):
        self.samples = list(samples)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = s.patch_bgr
        if self.augment:
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0.7)
            if random.random() < 0.5:
                jitter = np.random.normal(0, 5.0, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = np.int64(s.binary_label)
        return torch.from_numpy(x), torch.tensor(y), {
            "layout_number": int(s.layout_number),
            "slot_id": int(s.slot_id),
            "slot_name": str(s.slot_name),
            "gt_label": str(s.gt_label),
        }


class TinyBinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
            img_bgr, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101
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


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"]))


def build_samples(
    metadata: Dict,
    images_dir: Path,
    patch_size: int,
    context_scale: float,
    min_half_frac: float,
    target_slot_ids: Sequence[int],
) -> List[Sample]:
    layouts = {int(item["layout_number"]): item for item in metadata["layouts"]}
    gh_slots = list(metadata["slots_by_earth"]["Great Hollow"])

    slot_id_set = {int(x) for x in target_slot_ids}
    selected_slots = []
    for s in gh_slots:
        if slot_id_set and int(s.get("slot_id", -1)) not in slot_id_set:
            continue
        allowed = {str(x) for x in s.get("allowed_types", [])}
        if allowed & POSITIVE_LABELS:
            selected_slots.append(s)

    samples: List[Sample] = []
    for layout_number, layout in layouts.items():
        if str(layout.get("shifting_earth")) != "Great Hollow":
            continue
        image_path = images_dir / f"layout_{layout_number}.png"
        if not image_path.exists():
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        if h != w:
            img = img[:side, :side].copy()
            h = w = side

        slot_defs = gh_slots
        labels = list(layout.get("slot_labels", []))
        if len(slot_defs) != len(labels):
            continue

        by_slot_id = {int(s["slot_id"]): (s, str(labels[i])) for i, s in enumerate(slot_defs)}
        for s in selected_slots:
            sid = int(s["slot_id"])
            slot_def, gt = by_slot_id[sid]
            if gt not in POSITIVE_LABELS and gt not in NEGATIVE_LABELS:
                continue
            binary = 0 if gt in NEGATIVE_LABELS else 1
            cx = int(round(float(slot_def["x_norm"]) * w))
            cy = int(round(float(slot_def["y_norm"]) * h))
            half = int(round(max(float(slot_def["radius_norm"]) * side * context_scale, min_half_frac * side)))
            half = max(18, half)
            patch = reflect_pad_extract(img, cx, cy, half, patch_size)
            samples.append(
                Sample(
                    image_path=image_path,
                    layout_number=int(layout_number),
                    slot_id=sid,
                    slot_name=str(slot_def["slot_name"]),
                    gt_label=gt,
                    binary_label=int(binary),
                    patch_bgr=patch,
                )
            )
    if not samples:
        raise RuntimeError("No samples built.")
    return samples


def split_by_layout(samples: Sequence[Sample], train_ratio: float, val_ratio: float, seed: int):
    layouts = sorted({int(s.layout_number) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(layouts)
    n = len(layouts)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, min(n_val, n - 2))
        n_train = max(1, n - n_val - 1)
    train_ids = set(layouts[:n_train])
    val_ids = set(layouts[n_train:n_train + n_val])
    test_ids = set(layouts[n_train + n_val:])
    if not test_ids:
        test_ids = {layouts[-1]}
        train_ids.discard(layouts[-1])
    train = [s for s in samples if s.layout_number in train_ids]
    val = [s for s in samples if s.layout_number in val_ids]
    test = [s for s in samples if s.layout_number in test_ids]
    return train, val, test


def build_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    counts = Counter(int(s.binary_label) for s in samples)
    weights = [1.0 / max(1, counts[int(s.binary_label)]) for s in samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def class_weights_from_samples(samples: Sequence[Sample]) -> torch.Tensor:
    counts = Counter(int(s.binary_label) for s in samples)
    total = float(sum(counts.values()))
    w0 = total / (2.0 * max(1, float(counts.get(0, 0))))
    w1 = total / (2.0 * max(1, float(counts.get(1, 0))))
    return torch.tensor([w0, w1], dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    total = 0
    correct = 0
    conf = np.zeros((2, 2), dtype=np.int64)  # true/pred : empty/church_like
    empty_probs_empty = []
    empty_probs_pos = []
    for xb, yb, _meta in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        total += int(yb.numel())
        correct += int((pred == yb).sum().item())
        p_empty = probs[:, 0].cpu().numpy()
        y_np = yb.cpu().numpy()
        for i in range(len(y_np)):
            t = int(y_np[i]); p = int(pred[i].item())
            conf[t, p] += 1
            if t == 0:
                empty_probs_empty.append(float(p_empty[i]))
            else:
                empty_probs_pos.append(float(p_empty[i]))
    acc = float(correct / max(1, total))
    mean_empty_on_empty = float(np.mean(empty_probs_empty)) if empty_probs_empty else 0.0
    mean_empty_on_pos = float(np.mean(empty_probs_pos)) if empty_probs_pos else 0.0
    separation = float(mean_empty_on_empty - mean_empty_on_pos)
    return {
        "count": int(total),
        "accuracy": acc,
        "confusion_true_empty_pos_pred_empty_pos": conf.tolist(),
        "mean_p_empty_when_true_empty": mean_empty_on_empty,
        "mean_p_empty_when_true_church_like": mean_empty_on_pos,
        "empty_confidence_separation": separation,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(Path(args.package))
    samples = build_samples(
        metadata=metadata,
        images_dir=Path(args.images_dir),
        patch_size=int(args.patch_size),
        context_scale=float(args.context_scale),
        min_half_frac=float(args.min_half_frac),
        target_slot_ids=args.target_slot_ids,
    )
    train_s, val_s, test_s = split_by_layout(samples, args.train_ratio, args.val_ratio, args.seed)

    train_ds = PatchDataset(train_s, augment=True)
    val_ds = PatchDataset(val_s, augment=False)
    test_ds = PatchDataset(test_s, augment=False)

    sampler = build_sampler(train_s) if args.balance_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = choose_device(args.device)
    model = TinyBinaryCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = class_weights_from_samples(train_s).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_state = None
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, _meta in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_loader, device)
        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:02d}/{args.epochs} val_acc={val_acc:.4f} "
            f"sep={float(val_metrics['empty_confidence_separation']):.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    model_path = out_dir / "gh_empty_vs_churchrise_test_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_map": {"empty": 0, "church_like": 1},
            "patch_size": int(args.patch_size),
            "context_scale": float(args.context_scale),
            "min_half_frac": float(args.min_half_frac),
        },
        model_path,
    )

    report = {
        "counts": {
            "all": len(samples),
            "train": len(train_s),
            "val": len(val_s),
            "test": len(test_s),
            "labels_all": dict(Counter(int(s.binary_label) for s in samples)),
        },
        "slot_ids_included": sorted({int(s.slot_id) for s in samples}),
        "config": {
            "package": str(args.package),
            "images_dir": str(args.images_dir),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patch_size": int(args.patch_size),
            "context_scale": float(args.context_scale),
            "min_half_frac": float(args.min_half_frac),
            "balance_sampler": bool(args.balance_sampler),
            "seed": int(args.seed),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
    }

    report_path = out_dir / "gh_empty_vs_churchrise_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved model: {model_path}")
    print(f"saved report: {report_path}")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimental Great Hollow binary slot model: empty vs church_like (Church/Great Church/Sorcerer's Rise)."
    )
    p.add_argument("--package", type=Path, default=Path("nightreign_geom_proto_v4_slots.npz"))
    p.add_argument("--images-dir", type=Path, default=Path("renders/recognizer_no_text"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gh_empty_vs_churchrise_test"))
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--context-scale", type=float, default=1.8)
    p.add_argument("--min-half-frac", type=float, default=0.075)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--balance-sampler", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--target-slot-ids", type=int, nargs="+", default=[3, 4, 6])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
