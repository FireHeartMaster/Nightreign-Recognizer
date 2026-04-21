import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 200
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class PalaceStatusCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
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
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def discover_class_names(data_dir: Path) -> List[str]:
    classes: List[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir():
            has_any = any(f.is_file() and f.suffix.lower() in VALID_EXTS for f in p.iterdir())
            if has_any:
                classes.append(p.name)
    return classes


def collect_samples(data_dir: Path, class_names: Sequence[str]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for idx, class_name in enumerate(class_names):
        cls_dir = data_dir / class_name
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                samples.append((p, idx))
    return samples


def preprocess_bgr_200(img_bgr: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image received.")
    if img_bgr.shape[0] != image_size or img_bgr.shape[1] != image_size:
        img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    return x


def save_checkpoint(
    out_path: Path,
    model: nn.Module,
    class_names: Sequence[str],
    image_size: int,
    meta: Dict[str, object],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": int(image_size),
        "meta": dict(meta),
    }
    torch.save(payload, out_path)


def load_checkpoint(model_path: Path, device: torch.device) -> Tuple[nn.Module, List[str], int, Dict[str, object]]:
    raw = torch.load(model_path, map_location=device)
    class_names = [str(x) for x in raw["class_names"]]
    image_size = int(raw.get("image_size", IMAGE_SIZE))
    model = PalaceStatusCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, class_names, image_size, dict(raw.get("meta", {}))


def infer_logits(
    model: nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    x = preprocess_bgr_200(img, image_size=image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.squeeze(0).detach().cpu()


def infer_proba(
    model: nn.Module,
    image_path: Path,
    class_names: Sequence[str],
    image_size: int,
    device: torch.device,
) -> Dict[str, object]:
    logits = infer_logits(model, image_path=image_path, image_size=image_size, device=device)
    probs = F.softmax(logits, dim=0).numpy()
    ranking = sorted(
        [{"label": class_names[i], "probability": float(probs[i])} for i in range(len(class_names))],
        key=lambda x: x["probability"],
        reverse=True,
    )
    return {
        "image": str(image_path),
        "predicted_label": ranking[0]["label"],
        "confidence": float(ranking[0]["probability"]),
        "ranking": ranking,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
