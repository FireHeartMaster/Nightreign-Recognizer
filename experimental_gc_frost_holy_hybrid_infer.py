import argparse
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

import nightreign_map_recognizer_gui_ai_v37_completion_toast_overlay as gui


@dataclass
class TrainSample:
    image_path: Path
    label: str  # Frost | Holy


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


def crop_with_pad(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    half = int(size) // 2
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + int(size), y0 + int(size)
    h, w = img.shape[:2]
    xx0, yy0 = max(0, x0), max(0, y0)
    xx1, yy1 = min(w, x1), min(h, y1)
    patch = img[yy0:yy1, xx0:xx1].copy()
    if patch.size == 0:
        raise RuntimeError(f"Empty crop at ({cx},{cy}) size={size}")
    top = yy0 - y0
    left = xx0 - x0
    bottom = y1 - yy1
    right = x1 - xx1
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return patch


def softmax2(a: float, b: float, temp: float = 0.10) -> Tuple[float, float]:
    t = max(1e-6, float(temp))
    xa = float(a) / t
    xb = float(b) / t
    m = max(xa, xb)
    ea = math.exp(xa - m)
    eb = math.exp(xb - m)
    z = ea + eb + 1e-12
    return float(ea / z), float(eb / z)


def sharpen(p: float, scale: float = 3.8) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    logit = math.log(p / (1.0 - p))
    z = float(scale) * logit
    return float(1.0 / (1.0 + math.exp(-z)))


def color_desc(patch_bgr: np.ndarray) -> Dict[str, np.ndarray | float]:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    mask = ((sch > 40) & (vch > 40)).astype(np.uint8)
    if int(mask.sum()) < 8:
        mask = np.ones_like(mask, dtype=np.uint8)

    hist = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    hist /= float(hist.sum() + 1e-6)

    hh = hch.astype(np.float32)
    ss = sch.astype(np.float32)
    vv = vch.astype(np.float32)
    m = mask.astype(np.float32)
    w = float(m.sum() + 1e-6)
    mean_h = float((hh * m).sum() / w)
    mean_s = float((ss * m).sum() / w)
    mean_v = float((vv * m).sum() / w)

    return {"hist": hist, "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v}


def color_score(desc: Dict[str, np.ndarray | float], proto: Dict[str, np.ndarray | float]) -> float:
    hist_corr = float(
        cv2.compareHist(
            np.asarray(desc["hist"], dtype=np.float32),
            np.asarray(proto["hist"], dtype=np.float32),
            cv2.HISTCMP_CORREL,
        )
    )
    dh = abs(float(desc["mean_h"]) - float(proto["mean_h"]))
    dh = min(dh, 180.0 - dh) / 90.0  # ~[0,2] normalized to circle half
    ds = abs(float(desc["mean_s"]) - float(proto["mean_s"])) / 255.0
    dv = abs(float(desc["mean_v"]) - float(proto["mean_v"])) / 255.0
    hsv_close = 1.0 - min(1.0, 0.55 * dh + 0.30 * ds + 0.15 * dv)
    return float(0.82 * hist_corr + 0.18 * hsv_close)


def load_training_samples(pattern_dir: Path, image_dir: Path) -> List[TrainSample]:
    out: List[TrainSample] = []
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
                out.append(TrainSample(img_path, "Frost"))
                break
            if t == "Status_Holy":
                out.append(TrainSample(img_path, "Holy"))
                break
    return out


def _panel_box_from_model_meta(
    panel_bgr: np.ndarray,
    model_meta: dict,
    anchor_xy_legacy: Tuple[int, int] = (622, 1063),
) -> Tuple[int, int, int]:
    h, w = panel_bgr.shape[:2]
    mode = str(model_meta.get("box_mode", "legacy_anchor"))
    if mode == "proportional_bl_1024":
        x_bl_1024 = float(model_meta["x_bl_1024"])
        y_bl_1024 = float(model_meta["y_bl_1024"])
        crop_size_1024 = int(model_meta["crop_size_1024"])
        cx = int(round((x_bl_1024 / 1024.0) * w))
        cy = int(round((1.0 - (y_bl_1024 / 1024.0)) * h))
        side = max(12, int(round(crop_size_1024 * min(w, h) / 1024.0)))
        return cx, cy, side

    crop_size = int(model_meta["crop_size"])
    cx, cy = int(anchor_xy_legacy[0]), int(anchor_xy_legacy[1])
    return cx, cy, crop_size


def build_color_prototypes(
    samples: List[TrainSample],
    model_meta: dict,
    anchor_xy_legacy: Tuple[int, int] = (622, 1063),
) -> Dict[str, Dict[str, np.ndarray | float]]:
    by_lbl: Dict[str, List[Dict[str, np.ndarray | float]]] = {"Frost": [], "Holy": []}
    for s in samples:
        img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cx, cy, side = _panel_box_from_model_meta(img, model_meta, anchor_xy_legacy=anchor_xy_legacy)
        patch = crop_with_pad(img, cx, cy, side)
        by_lbl[s.label].append(color_desc(patch))

    protos: Dict[str, Dict[str, np.ndarray | float]] = {}
    for lbl in ("Frost", "Holy"):
        xs = by_lbl.get(lbl, [])
        if not xs:
            raise RuntimeError(f"No samples for class: {lbl}")
        hist = np.mean(np.stack([np.asarray(x["hist"], dtype=np.float32) for x in xs], axis=0), axis=0).astype(np.float32)
        hist /= float(hist.sum() + 1e-6)
        protos[lbl] = {
            "hist": hist,
            "mean_h": float(np.mean([float(x["mean_h"]) for x in xs])),
            "mean_s": float(np.mean([float(x["mean_s"]) for x in xs])),
            "mean_v": float(np.mean([float(x["mean_v"]) for x in xs])),
        }
    return protos


def load_tiny_model(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    model = TinyStatusCNN(num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label_map = ckpt.get("label_map", {"Frost": 0, "Holy": 1})
    id_to_label = {int(v): str(k) for k, v in label_map.items()}
    out_size = int(ckpt.get("out_size", 32))
    if "crop_size_1024" in ckpt:
        model_meta = {
            "box_mode": "proportional_bl_1024",
            "crop_size_1024": int(ckpt["crop_size_1024"]),
            "x_bl_1024": float(ckpt.get("x_bl_1024", 504.0)),
            "y_bl_1024": float(ckpt.get("y_bl_1024", 182.0)),
        }
    else:
        model_meta = {
            "box_mode": "legacy_anchor",
            "crop_size": int(ckpt.get("crop_size", 28)),
        }
    return model, id_to_label, out_size, model_meta


def tiny_probs(model: nn.Module, id_to_label: Dict[int, str], patch_bgr: np.ndarray, out_size: int) -> Dict[str, float]:
    x = cv2.resize(patch_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    xt = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {id_to_label[i]: float(probs[i]) for i in range(len(probs))}


def extract_panel_1536(img_bgr: np.ndarray, backend, ctx) -> Tuple[np.ndarray, str]:
    def detect_map_circle_compat(local_img: np.ndarray):
        region = getattr(backend, "DEFAULT_HOUGH_REGION", "right_half")
        scale = float(getattr(backend, "DEFAULT_HOUGH_SCALE", 1.0))
        try:
            return backend.detect_map_circle(local_img, region, scale, ctx)
        except TypeError:
            try:
                return backend.detect_map_circle(local_img, region, scale)
            except TypeError:
                return backend.detect_map_circle(local_img)

    def refine_panel_box_compat(local_img: np.ndarray, circle_obj):
        panel_scale = float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))
        search_radius = int(getattr(backend, "DEFAULT_SEARCH_RADIUS", 80))
        pad = int(getattr(backend, "DEFAULT_PAD", 0))
        try:
            return backend.refine_panel_box(local_img, circle_obj, panel_scale, search_radius, pad, ctx)
        except TypeError:
            try:
                return backend.refine_panel_box(local_img, circle_obj, panel_scale, search_radius, pad)
            except TypeError:
                return backend.refine_panel_box(local_img, circle_obj)

    def normalize_panel_compat(raw_panel: np.ndarray):
        fn = getattr(backend, "normalize_panel_for_slot_recognition", None)
        if fn is None:
            fn = getattr(backend, "normalize_panel_for_recognition", None)
        if fn is None:
            raise RuntimeError("Backend has no panel normalization function.")
        return fn(raw_panel)

    circle, crop_method = detect_map_circle_compat(img_bgr)
    if crop_method == "blue_ring":
        crop_box = backend.clip_box(
            backend.estimate_panel_box(circle, float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))),
            img_bgr.shape[1],
            img_bgr.shape[0],
        )
    else:
        crop_box = refine_panel_box_compat(img_bgr, circle)
    raw_panel = img_bgr[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
    panel = normalize_panel_compat(raw_panel)
    return panel, str(crop_method)


def infer_hybrid_for_files(
    files: List[Path],
    model_path: Path,
    pattern_dir: Path,
    image_dir: Path,
    out_dir: Path,
    anchor_xy: Tuple[int, int] = (622, 1063),
    alpha_model: float = 0.30,
    sharp_scale: float = 3.8,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, id_to_label, out_size, model_meta = load_tiny_model(model_path)
    train_samples = load_training_samples(pattern_dir, image_dir)
    protos = build_color_prototypes(train_samples, model_meta, anchor_xy_legacy=anchor_xy)

    logs: List[str] = []
    ctx = gui.RunContext(9100, threading.Event(), lambda run_id, msg: logs.append(str(msg)))
    backend = gui.load_proto_backend_module(ctx)
    tuned_cfg = gui.load_tuned_config(ctx)
    gui.apply_config_to_module(backend, tuned_cfg)

    results: List[dict] = []
    for fp in files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            results.append({"file": str(fp), "error": "could_not_read"})
            continue
        # Always use the direct normalized panel for inference (no extra/further crop),
        # matching how the training panels are represented.
        panel, crop_method = extract_panel_1536(img, backend, ctx)
        cx, cy, side = _panel_box_from_model_meta(panel, model_meta, anchor_xy_legacy=anchor_xy)
        patch = crop_with_pad(panel, cx, cy, side)

        probs_model = tiny_probs(model, id_to_label, patch, out_size)
        desc = color_desc(patch)
        score_frost = color_score(desc, protos["Frost"])
        score_holy = color_score(desc, protos["Holy"])
        color_pf, color_ph = softmax2(score_frost, score_holy, temp=0.08)

        m_pf = float(probs_model.get("Frost", 0.5))
        m_ph = float(probs_model.get("Holy", 0.5))
        blend_pf = (1.0 - float(alpha_model)) * color_pf + float(alpha_model) * m_pf
        blend_ph = (1.0 - float(alpha_model)) * color_ph + float(alpha_model) * m_ph

        if blend_pf + blend_ph <= 1e-12:
            blend_pf, blend_ph = 0.5, 0.5
        else:
            z = float(blend_pf + blend_ph)
            blend_pf, blend_ph = float(blend_pf / z), float(blend_ph / z)

        frost_sharp = sharpen(blend_pf, scale=sharp_scale)
        holy_sharp = 1.0 - frost_sharp
        pred = "Frost" if frost_sharp >= 0.5 else "Holy"
        margin = abs(frost_sharp - holy_sharp)

        patch_file = out_dir / f"{fp.stem}__gc_frost_holy_hybrid_crop.png"
        cv2.imwrite(str(patch_file), patch)

        results.append(
            {
                "file": str(fp),
                "crop_method": crop_method,
                "panel_shape": [int(panel.shape[1]), int(panel.shape[0])],
                "anchor_xy": [int(cx), int(cy)],
                "crop_size": int(side),
                "prediction": pred,
                "margin": float(margin),
                "model_probs": {"Frost": m_pf, "Holy": m_ph},
                "color_scores": {"Frost": float(score_frost), "Holy": float(score_holy)},
                "color_probs": {"Frost": float(color_pf), "Holy": float(color_ph)},
                "hybrid_probs": {"Frost": float(frost_sharp), "Holy": float(holy_sharp)},
                "crop_file": str(patch_file),
                "config": {
                    "alpha_model": float(alpha_model),
                    "sharp_scale": float(sharp_scale),
                    "model_meta": model_meta,
                },
            }
        )
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Test-only hybrid Frost/Holy classifier on screenshots.")
    p.add_argument("--files", nargs="+", required=True, help="Screenshot files to test.")
    p.add_argument("--model", type=Path, default=Path("tmp_gc_frost_holy_test/gc_frost_holy_test_model.pt"))
    p.add_argument("--pattern-dir", type=Path, default=Path("renders_new_under/no_text"))
    p.add_argument("--image-dir", type=Path, default=Path("renders/recognizer_no_text_under"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gc_frost_holy_hybrid_test"))
    p.add_argument("--alpha-model", type=float, default=0.30)
    p.add_argument("--sharp-scale", type=float, default=3.8)
    return p.parse_args()


def main():
    args = parse_args()
    files = [Path(x) for x in args.files]
    results = infer_hybrid_for_files(
        files=files,
        model_path=args.model,
        pattern_dir=args.pattern_dir,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        alpha_model=float(args.alpha_model),
        sharp_scale=float(args.sharp_scale),
    )
    out_json = args.out_dir / "hybrid_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
