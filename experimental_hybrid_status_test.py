import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import icon_recognizer_v4


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


def crop_with_padding(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + size)
    y1 = min(h, y + size)
    patch = img[y0:y1, x0:x1].copy()
    if patch.size == 0:
        raise RuntimeError(f"Empty crop at x={x}, y={y}, size={size}")
    top = y0 - y
    left = x0 - x
    bottom = (y + size) - y1
    right = (x + size) - x1
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return patch


def softmax_two(a: float, b: float, temp: float = 0.08):
    t = max(1e-6, float(temp))
    xa = float(a) / t
    xb = float(b) / t
    m = max(xa, xb)
    ea = math.exp(xa - m)
    eb = math.exp(xb - m)
    z = ea + eb + 1e-12
    return float(ea / z), float(eb / z)


def blend_probs(icon_p: float, model_p: float, alpha_model: float = 0.65):
    a = float(np.clip(alpha_model, 0.0, 1.0))
    return float((1.0 - a) * icon_p + a * model_p)


def sharpen_binary_prob(p: float, logit_scale: float = 2.8):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    logit = math.log(p / (1.0 - p))
    z = float(logit_scale) * logit
    return float(1.0 / (1.0 + math.exp(-z)))


def load_tiny_model(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    model = TinyStatusCNN(num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label_map = ckpt["label_map"]
    crop_size = int(ckpt["crop_size"])
    out_size = int(ckpt["out_size"])
    id_to_label = {int(v): str(k) for k, v in label_map.items()}
    return model, id_to_label, crop_size, out_size


def infer_tiny_model(model: nn.Module, id_to_label: dict, patch_bgr: np.ndarray, out_size: int):
    x = cv2.resize(patch_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    xt = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs_map = {id_to_label[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = int(np.argmax(probs))
    pred = id_to_label[pred_idx]
    return pred, probs_map


def extract_icon_probs(result_entry: dict, label_a: str, label_b: str):
    ranked = {str(r["label"]): float(r["score"]) for r in result_entry.get("ranked", [])}
    sa = ranked.get(label_a, float("-inf"))
    sb = ranked.get(label_b, float("-inf"))
    pa, pb = softmax_two(sa, sb, temp=0.08)
    return {label_a: pa, label_b: pb}, {label_a: sa, label_b: sb}


def classify_hybrid_for_query(
    ref1: Path,
    ref2: Path,
    query_panel: Path,
    palace_model_path: Path,
    fortress_model_path: Path,
    church_model_path: Path,
):
    icon = icon_recognizer_v4.recognize_markers(str(ref1), str(ref2), str(query_panel))
    img = cv2.imread(str(query_panel), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read query panel: {query_panel}")

    p_model, p_id_to_label, p_crop, p_out = load_tiny_model(palace_model_path)
    f_model, f_id_to_label, f_crop, f_out = load_tiny_model(fortress_model_path)
    c_model, c_id_to_label, c_crop, c_out = load_tiny_model(church_model_path)

    # Palace: Magic/Fire
    p_box = icon["top_left"]["box"]
    p_patch = crop_with_padding(img, int(p_box["x"]), int(p_box["y"]), int(p_box["w"]))
    _p_pred, p_probs = infer_tiny_model(p_model, p_id_to_label, p_patch, p_out)
    p_icon_probs, p_icon_scores = extract_icon_probs(icon["top_left"], "Magic", "Fire")
    p_magic = blend_probs(p_icon_probs["Magic"], p_probs.get("Magic", 0.5), alpha_model=0.65)
    p_magic_sharp = sharpen_binary_prob(p_magic, logit_scale=2.8)
    p_fire_sharp = 1.0 - p_magic_sharp
    p_label = "Magic" if p_magic_sharp >= 0.5 else "Fire"
    p_margin = abs(p_magic_sharp - p_fire_sharp)

    # Fortress: Rot/Holy
    f_box = icon["bottom_right"]["box"]
    f_patch = crop_with_padding(img, int(f_box["x"]), int(f_box["y"]), int(f_box["w"]))
    _f_pred, f_probs = infer_tiny_model(f_model, f_id_to_label, f_patch, f_out)
    f_icon_probs, f_icon_scores = extract_icon_probs(icon["bottom_right"], "Rot", "Holy")
    f_rot = blend_probs(f_icon_probs["Rot"], f_probs.get("Rot", 0.5), alpha_model=0.65)
    f_rot_sharp = sharpen_binary_prob(f_rot, logit_scale=2.8)
    f_holy_sharp = 1.0 - f_rot_sharp
    f_label = "Rot" if f_rot_sharp >= 0.5 else "Holy"
    f_margin = abs(f_rot_sharp - f_holy_sharp)

    # Great Church: Lightning/Sleep
    c_box = icon["great_church_top_right"]["box"]
    c_patch = crop_with_padding(img, int(c_box["x"]), int(c_box["y"]), int(c_box["w"]))
    _c_pred, c_probs = infer_tiny_model(c_model, c_id_to_label, c_patch, c_out)
    c_icon_probs, c_icon_scores = extract_icon_probs(icon["great_church_top_right"], "Lightning", "Sleep")
    c_light = blend_probs(c_icon_probs["Lightning"], c_probs.get("Lightning", 0.5), alpha_model=0.70)
    c_light_sharp = sharpen_binary_prob(c_light, logit_scale=3.2)
    c_sleep_sharp = 1.0 - c_light_sharp
    c_label = "Lightning" if c_light_sharp >= 0.5 else "Sleep"
    c_margin = abs(c_light_sharp - c_sleep_sharp)

    return {
        "query_panel": str(query_panel),
        "icon_recognizer": {
            "palace": icon["top_left"],
            "fortress": icon["bottom_right"],
            "great_church_top_right": icon["great_church_top_right"],
        },
        "hybrid": {
            "palace": {
                "label": p_label,
                "probs_sharpened": {"Magic": float(p_magic_sharp), "Fire": float(p_fire_sharp)},
                "margin": float(p_margin),
                "icon_scores": p_icon_scores,
                "icon_probs": p_icon_probs,
                "model_probs": {"Magic": float(p_probs.get("Magic", 0.5)), "Fire": float(p_probs.get("Fire", 0.5))},
            },
            "fortress": {
                "label": f_label,
                "probs_sharpened": {"Rot": float(f_rot_sharp), "Holy": float(f_holy_sharp)},
                "margin": float(f_margin),
                "icon_scores": f_icon_scores,
                "icon_probs": f_icon_probs,
                "model_probs": {"Rot": float(f_probs.get("Rot", 0.5)), "Holy": float(f_probs.get("Holy", 0.5))},
            },
            "great_church_top_right": {
                "label": c_label,
                "probs_sharpened": {"Lightning": float(c_light_sharp), "Sleep": float(c_sleep_sharp)},
                "margin": float(c_margin),
                "icon_scores": c_icon_scores,
                "icon_probs": c_icon_probs,
                "model_probs": {"Lightning": float(c_probs.get("Lightning", 0.5)), "Sleep": float(c_probs.get("Sleep", 0.5))},
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description="Test-only hybrid status classifier (icon_recognizer + tiny models).")
    p.add_argument("--ref1", type=Path, required=True)
    p.add_argument("--ref2", type=Path, required=True)
    p.add_argument("--queries", nargs="+", required=True, help="List of panel images.")
    p.add_argument("--palace-model", type=Path, default=Path("tmp_palace_magic_fire_test/palace_magic_fire_test_model.pt"))
    p.add_argument("--fortress-model", type=Path, default=Path("tmp_fortress_rot_holy_test/fortress_rot_holy_test_model.pt"))
    p.add_argument("--church-model", type=Path, default=Path("tmp_gc_sleep_lightning_test/gc_sleep_lightning_test_model.pt"))
    p.add_argument("--out-json", type=Path, default=Path("tmp_hybrid_status_test/hybrid_results.json"))
    args = p.parse_args()

    out = []
    for q in args.queries:
        out.append(
            classify_hybrid_for_query(
                ref1=args.ref1,
                ref2=args.ref2,
                query_panel=Path(q),
                palace_model_path=args.palace_model,
                fortress_model_path=args.fortress_model,
                church_model_path=args.church_model,
            )
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_json}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
