import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Sample:
    image_path: Path
    x: int
    y: int
    label: int  # 0=Frost, 1=Holy
    pattern_id: int


class CropDataset(Dataset):
    def __init__(self, samples: List[Sample], crop_size: int = 28, out_size: int = 32, augment: bool = False):
        self.samples = samples
        self.crop_size = int(crop_size)
        self.out_size = int(out_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _crop_with_pad(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
        half = size // 2
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + size, y0 + size
        h, w = img.shape[:2]
        xx0, yy0 = max(0, x0), max(0, y0)
        xx1, yy1 = min(w, x1), min(h, y1)
        patch = img[yy0:yy1, xx0:xx1].copy()
        top = yy0 - y0
        left = xx0 - x0
        bottom = y1 - yy1
        right = x1 - xx1
        if top or left or bottom or right:
            patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        return patch

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {s.image_path}")

        patch = self._crop_with_pad(img, int(s.x), int(s.y), self.crop_size)
        patch = cv2.resize(patch, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() < 0.5:
                patch = cv2.GaussianBlur(patch, (3, 3), 0.8)
            if random.random() < 0.5:
                jitter = np.random.normal(0, 5.0, patch.shape).astype(np.float32)
                patch = np.clip(patch.astype(np.float32) + jitter, 0, 255).astype(np.uint8)

        x = patch.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = np.int64(s.label)
        return torch.from_numpy(x), torch.tensor(y)


class TinyStatusCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_samples(pattern_dir: Path, image_dir: Path) -> List[Sample]:
    out: List[Sample] = []
    for p in sorted(pattern_dir.glob("pattern_*_Great Hollow_under.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = int(obj.get("patternId", -1))
        if pid < 0:
            continue
        img_path = image_dir / f"layout_{pid}.png"
        if not img_path.exists():
            continue

        for lm in obj.get("landmarks", []) or []:
            t = str(lm.get("type", ""))
            if t == "Status_Frost":
                out.append(Sample(img_path, int(round(float(lm["x"]))), int(round(float(lm["y"]))), 0, pid))
                break
            if t == "Status_Holy":
                out.append(Sample(img_path, int(round(float(lm["x"]))), int(round(float(lm["y"]))), 1, pid))
                break
    return out


def stratified_split(samples: List[Sample], val_ratio: float = 0.3, seed: int = 42) -> Tuple[List[Sample], List[Sample]]:
    random.seed(seed)
    by_label = {0: [], 1: []}
    for s in samples:
        by_label[int(s.label)].append(s)
    train, val = [], []
    for lbl in [0, 1]:
        xs = by_label[lbl]
        random.shuffle(xs)
        cut = int(round(len(xs) * (1.0 - val_ratio)))
        train.extend(xs[:cut])
        val.extend(xs[cut:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def eval_model(model: nn.Module, dl: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    conf = np.zeros((2, 2), dtype=np.int64)  # true, pred ; Frost/Holy
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            total += int(yb.numel())
            correct += int((pred == yb).sum().item())
            for t, p in zip(yb.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
                conf[int(t), int(p)] += 1
    return {
        "count": int(total),
        "accuracy": float(correct / max(1, total)),
        "confusion_true_frost_holy_pred_frost_holy": conf.tolist(),
    }


def train(args) -> None:
    pattern_dir = Path(args.pattern_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(pattern_dir, image_dir)
    if not samples:
        raise RuntimeError("No Frost/Holy samples found.")

    train_s, val_s = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyStatusCNN(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    ds_train = CropDataset(train_s, crop_size=args.crop_size, out_size=args.out_size, augment=True)
    ds_val = CropDataset(val_s, crop_size=args.crop_size, out_size=args.out_size, augment=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach().cpu()) * int(yb.numel())
            n += int(yb.numel())
        train_loss = run_loss / max(1, n)
        val_metrics = eval_model(model, dl_val, device)
        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = eval_model(model, dl_train, device)
    final_val = eval_model(model, dl_val, device)

    model_path = out_dir / "gc_frost_holy_test_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_map": {"Frost": 0, "Holy": 1},
            "crop_size": int(args.crop_size),
            "out_size": int(args.out_size),
        },
        model_path,
    )

    report = {
        "train_count": len(train_s),
        "val_count": len(val_s),
        "all_count": len(samples),
        "train_metrics": final_train,
        "val_metrics": final_val,
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "crop_size": int(args.crop_size),
            "out_size": int(args.out_size),
            "seed": int(args.seed),
        },
        "model_path": str(model_path),
    }

    report_path = out_dir / "gc_frost_holy_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved model: {model_path}")
    print(f"saved report: {report_path}")
    print(json.dumps(report["val_metrics"], indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Experimental Great Church Frost/Holy classifier test.")
    p.add_argument("--pattern-dir", type=Path, default=Path("renders_new_under/no_text"))
    p.add_argument("--image-dir", type=Path, default=Path("renders/recognizer_no_text_under"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gc_frost_holy_test"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop-size", type=int, default=28)
    p.add_argument("--out-size", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

