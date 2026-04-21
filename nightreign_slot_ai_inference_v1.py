#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn

IMAGE_W = 1536.0
IMAGE_H = 1536.0
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0
EVERGAOL_ICON_PATH = Path("POI_icons") / "Evergaol.png"
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
EVERGAOL_TEMPLATE_CACHE: Optional[List[dict]] = None
GREAT_HOLLOW_STATUS_ICON_PATHS = {
    "Status_Poison": Path("POI_icons") / "Status_Poison.png",
    "Status_Bleed": Path("POI_icons") / "Status_Bleed.png",
    "Status_Sleep": Path("POI_icons") / "Status_Sleep.png",
    "Status_Lightning": Path("POI_icons") / "Status_Lightning.png",
    "Status_Rot": Path("POI_icons") / "Status_Rot.png",
    "Status_Holy": Path("POI_icons") / "Status_Holy.png",
    "Status_Fire": Path("POI_icons") / "Status_Fire.png",
    "Status_Magic": Path("POI_icons") / "Status_Magic.png",
}
GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE = {
    "Ruins": ("Status_Poison", "Status_Bleed"),
    "Great Church": ("Status_Sleep", "Status_Lightning"),
    "Fortress": ("Status_Rot", "Status_Holy"),
    "Palace": ("Status_Fire", "Status_Magic"),
}
GREAT_HOLLOW_STATUS_PATTERN_DIRS = (
    Path("renders_new") / "no_text",
    Path("renders_new") / "with_text",
)
GREAT_HOLLOW_FIXED_STATUS_REFS = {
    "ref1": (
        Path("layout_1007.png"),
        Path("renders") / "recognizer_no_text" / "layout_1007.png",
    ),
    "ref2": (
        Path("layout_1005.png"),
        Path("renders") / "recognizer_no_text" / "layout_1005.png",
    ),
}
GREAT_HOLLOW_FIXED_STATUS_CONFIG = {
    "Palace": {
        "xy_norm": (0.328125, 0.28046875),
        "allowed": ("Status_Fire", "Status_Magic"),
        "ref1_label": "Status_Fire",
        "ref2_label": "Status_Magic",
    },
    "Fortress": {
        "xy_norm": (0.68359375, 0.7015625),
        "allowed": ("Status_Rot", "Status_Holy"),
        "ref1_label": "Status_Rot",
        "ref2_label": "Status_Holy",
    },
}
GREAT_HOLLOW_PATTERNSET_DIR = Path("great_hollow_pattern_sets")
GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD = {
    "greatHollowEastFortMainModifier1": ("Fortress", "Status_Rot"),
    "greatHollowEastFortMainModifier2": ("Fortress", "Status_Holy"),
    "greatHollowWestFortMainModifier1": ("Palace", "Status_Fire"),
    "greatHollowWestFortMainModifier2": ("Palace", "Status_Magic"),
    # Great Hollow-specific status anchors beyond Palace/Fortress.
    # Mapping convention follows modifier index 1->first allowed status, 2->second allowed status.
    "greatHollowWestFortUnderModifier1": ("Ruins", "Status_Poison"),
    "greatHollowWestFortUnderModifier2": ("Ruins", "Status_Bleed"),
    "greatHollowWestFortLibraryModifier1": ("Great Church", "Status_Sleep"),
    "greatHollowWestFortLibraryModifier2": ("Great Church", "Status_Lightning"),
}
GREAT_HOLLOW_SYNTHETIC_WORLD_POINTS = {
    "Fortress": (734.399, 267.268),
    "Palace": (278.752, 805.636),
    "Ruins": (138.250, 392.670),
    "Great Church": (933.460, 548.590),
}
GREAT_HOLLOW_CROP_BORDER = 128.0
STATUS_RECOGNIZER_INPUT_SIZE = 1536
STATUS_RECOGNIZER_BORDER_CROP_PX = 256
GREAT_HOLLOW_STATUS_ICON_SOURCES = None
GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = None
GREAT_HOLLOW_LAYOUT_STATUS_CACHE = {}
GREAT_HOLLOW_STATUS_ANCHOR_CACHE = None
ICON_RECOGNIZER_V4_MODULE = None
ICON_RECOGNIZER_V4_LOAD_ATTEMPTED = False
_MAP_CLASSIFIER_CACHE: Optional[Dict[str, Any]] = None
USE_GREAT_HOLLOW_STATUS_PRIOR = True
EVERGAOL_MODEL_FILENAMES = [
    "evergaol_ai_test_output_no_text/evergaol_patch_model_v1.pt",
    "evergaol_ai_test_output/evergaol_patch_model_v1.pt",
]
_EVERGAOL_MODEL_CACHE = None

class SmallSlotCNN(nn.Module):
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
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test the trained slot AI model on a real image, a pre-cropped panel image, or a live monitor."
    )
    p.add_argument("--model", type=Path, required=True, help="Path to slot_patch_model_v1.pt")
    p.add_argument("--backend", type=Path, required=True, help="Path to the backend recognizer script")
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package")
    p.add_argument("--config-json", type=Path, default=None, help="Optional tuned config JSON to apply to the backend crop/earth stage")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Full screenshot image to process")
    source.add_argument("--panel-image", type=Path, help="Already-cropped square panel image to process")
    source.add_argument("--monitor", type=int, help="Capture this monitor index with mss")

    p.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    p.add_argument("--top", type=int, default=20, help="How many ranked layouts to print")
    p.add_argument("--confidence-threshold", type=float, default=0.55, help="Below this probability, label the slot as unknown")
    p.add_argument("--margin-threshold", type=float, default=0.08, help="If top1-top2 probability gap is below this, label the slot as unknown")
    p.add_argument("--debug", action="store_true", help="Save debug images")
    p.add_argument("--panel-out", type=Path, default=Path("ai_panel_used.png"))
    p.add_argument("--ai-debug-out", type=Path, default=Path("ai_slot_labels_debug.png"))
    p.add_argument("--evergaol-debug-out", type=Path, default=Path("ai_evergaol_debug.png"))
    p.add_argument("--json-out", type=Path, default=Path("ai_slot_inference_report.json"))
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_bgr(path: Optional[Path], img: np.ndarray) -> None:
    if path is not None:
        ensure_parent(path)
        cv2.imwrite(str(path), img)


def load_backend_module(path: Path):
    spec = importlib.util.spec_from_file_location("nightreign_ai_slot_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_evergaol_model_path() -> Optional[Path]:
    candidates: List[Path] = []
    for name in EVERGAOL_MODEL_FILENAMES:
        candidates.extend([
            Path.cwd() / name,
            Path(__file__).resolve().parent / name,
        ])
    seen = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def load_evergaol_model_bundle(device: torch.device):
    global _EVERGAOL_MODEL_CACHE
    if _EVERGAOL_MODEL_CACHE is not None:
        return _EVERGAOL_MODEL_CACHE
    model_path = find_evergaol_model_path()
    if model_path is None:
        _EVERGAOL_MODEL_CACHE = None
        return None
    bundle = torch.load(model_path, map_location=device)
    labels = list(bundle.get("labels", ["absent", "present"]))
    model = EvergaolPatchCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    _EVERGAOL_MODEL_CACHE = {
        "model_path": model_path,
        "model": model,
        "device": device,
        "labels": labels,
        "patch_size": int(bundle.get("patch_size", 96)),
        "context_scale": float(bundle.get("context_scale", 2.0)),
        "min_half_frac": float(bundle.get("min_half_frac", 0.070)),
    }
    return _EVERGAOL_MODEL_CACHE


def load_backend_package(module, package_path: Path):
    package_cls = getattr(module, "RecognitionPackage", None)
    if package_cls is None:
        package_cls = getattr(module, "SlotRecognitionPackage", None)
    if package_cls is None:
        raise AttributeError(
            f"Backend module '{module.__name__}' has neither RecognitionPackage nor SlotRecognitionPackage."
        )
    return package_cls(package_path)


def make_backend_ctx(module):
    ctx_cls = getattr(module, "RunContext", None)
    if ctx_cls is None:
        return None
    try:
        return ctx_cls(0, threading.Event(), lambda _run_id, message: log(message))
    except TypeError:
        return None


def detect_map_circle_compat(module, full_img: np.ndarray, ctx):
    if ctx is not None:
        try:
            return module.detect_map_circle(
                full_img,
                getattr(module, "DEFAULT_HOUGH_REGION", "right_half"),
                float(getattr(module, "DEFAULT_HOUGH_SCALE", 1.0)),
                ctx,
            )
        except TypeError:
            pass
    return module.detect_map_circle(full_img)


def refine_panel_box_compat(module, full_img: np.ndarray, circle, ctx):
    if ctx is not None:
        try:
            return module.refine_panel_box(
                full_img,
                circle,
                float(getattr(module, "DEFAULT_PANEL_SCALE", 1.12)),
                int(getattr(module, "DEFAULT_SEARCH_RADIUS", 80)),
                int(getattr(module, "DEFAULT_PAD", 0)),
                ctx,
            )
        except TypeError:
            pass
    return module.refine_panel_box(full_img, circle)


def refine_crop_box_by_earth_score_compat(module, full_img: np.ndarray, initial_box, package, ctx):
    fn = getattr(module, "refine_crop_box_by_earth_score", None)
    if fn is not None:
        try:
            return fn(full_img, initial_box, package)
        except TypeError:
            pass
    fn = getattr(module, "refine_crop_box_by_earth_score_slot", None)
    if fn is not None:
        if ctx is not None:
            try:
                return fn(full_img, initial_box, package, ctx)
            except TypeError:
                pass
        return fn(full_img, initial_box, package)
    raise AttributeError(
        f"Backend module '{module.__name__}' has neither refine_crop_box_by_earth_score nor refine_crop_box_by_earth_score_slot."
    )


def get_earth_secondary_threshold(module) -> float:
    value = getattr(module, "EARTH_SECONDARY_THRESHOLD", None)
    if value is None:
        value = getattr(module, "SLOT_EARTH_SECONDARY_THRESHOLD", 0.0)
    return float(value)


def normalize_panel_for_recognition_compat(module, raw_panel: np.ndarray) -> np.ndarray:
    fn = getattr(module, "normalize_panel_for_recognition", None)
    if fn is None:
        fn = getattr(module, "normalize_panel_for_slot_recognition", None)
    if fn is None:
        raise AttributeError(
            f"Backend module '{module.__name__}' has neither normalize_panel_for_recognition nor normalize_panel_for_slot_recognition."
        )
    return fn(raw_panel)


def estimate_global_translation_compat(module, panel_bgr: np.ndarray, package, earth_name: str):
    fn = getattr(module, "estimate_global_translation", None)
    if fn is None:
        fn = getattr(module, "estimate_global_translation_slot", None)
    if fn is None:
        raise AttributeError(
            f"Backend module '{module.__name__}' has neither estimate_global_translation nor estimate_global_translation_slot."
        )
    return fn(panel_bgr, package, earth_name)


def _normalize_earth_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _load_map_classifier_bundle() -> Optional[Dict[str, Any]]:
    global _MAP_CLASSIFIER_CACHE
    if _MAP_CLASSIFIER_CACHE is not None:
        return _MAP_CLASSIFIER_CACHE

    script_candidates = [
        Path.cwd() / "map_class_classifier.py",
        Path(__file__).resolve().parent / "map_class_classifier.py",
    ]
    model_candidates = [
        Path.cwd() / "map_class_model.pkl",
        Path(__file__).resolve().parent / "map_class_model.pkl",
    ]
    script_path = next((p for p in script_candidates if p.exists()), None)
    model_path = next((p for p in model_candidates if p.exists()), None)
    if script_path is None or model_path is None:
        _MAP_CLASSIFIER_CACHE = None
        return None

    spec = importlib.util.spec_from_file_location("nightreign_map_class_classifier_runtime", script_path)
    if spec is None or spec.loader is None:
        _MAP_CLASSIFIER_CACHE = None
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_model(str(model_path))
    _MAP_CLASSIFIER_CACHE = {
        "module": module,
        "model": model,
        "script_path": script_path,
        "model_path": model_path,
    }
    return _MAP_CLASSIFIER_CACHE


def _classify_shifting_earth_map_classifier(
    raw_panel_bgr: np.ndarray, package
) -> Optional[List[Tuple[str, float, Dict[str, float]]]]:
    bundle = _load_map_classifier_bundle()
    if bundle is None:
        return None

    module = bundle["module"]
    model = bundle["model"]
    try:
        descriptor = module.compute_descriptor(raw_panel_bgr)
        result = module.classify_descriptor(model, descriptor)
    except Exception:
        return None

    class_scores = result.get("class_scores", []) or []
    by_class = {
        _normalize_earth_name(item.get("label", "")): float(item.get("score", 0.0))
        for item in class_scores
    }
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for earth_name in package.earth_names:
        score = float(by_class.get(_normalize_earth_name(earth_name), 0.0))
        ranked.append(
            (
                earth_name,
                score,
                {
                    "source": "map_class_classifier",
                    "neighbors": result.get("neighbors", []),
                },
            )
        )
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def classify_shifting_earth_compat(module, raw_panel_bgr: np.ndarray, panel_bgr: np.ndarray, package):
    map_rank = _classify_shifting_earth_map_classifier(raw_panel_bgr, package)
    if map_rank:
        return map_rank

    fn = getattr(module, "classify_shifting_earth_ai", None)
    if fn is not None:
        try:
            return fn(raw_panel_bgr, panel_bgr, package, None)
        except TypeError:
            try:
                return fn(raw_panel_bgr, panel_bgr, package)
            except TypeError:
                return fn(panel_bgr, package)
    fn = getattr(module, "classify_shifting_earth", None)
    if fn is None:
        fn = getattr(module, "classify_shifting_earth_slot", None)
    if fn is None:
        raise AttributeError(
            f"Backend module '{module.__name__}' has neither classify_shifting_earth nor classify_shifting_earth_slot."
        )
    return fn(panel_bgr, package)


def clone_type_config(cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in cfg.items()}


def capture_base_state(module) -> Dict[str, Any]:
    return {
        "TYPE_CONFIG": clone_type_config(module.TYPE_CONFIG),
        "DEFAULT_TEMPLATE_SCALES": tuple(module.DEFAULT_TEMPLATE_SCALES),
        "CROP_REFINE_MARGIN_WEIGHT": float(module.CROP_REFINE_MARGIN_WEIGHT),
        "CROP_REFINE_BLACK_WEIGHT": float(module.CROP_REFINE_BLACK_WEIGHT),
        "CROP_REFINE_EDGE_WEIGHT": float(module.CROP_REFINE_EDGE_WEIGHT),
        "EARTH_SECONDARY_THRESHOLD": float(module.EARTH_SECONDARY_THRESHOLD),
        "classify_shifting_earth": module.classify_shifting_earth,
    }


def restore_base_state(module, base: Dict[str, Any]) -> None:
    module.TYPE_CONFIG = clone_type_config(base["TYPE_CONFIG"])
    module.DEFAULT_TEMPLATE_SCALES = tuple(base["DEFAULT_TEMPLATE_SCALES"])
    module.CROP_REFINE_MARGIN_WEIGHT = float(base["CROP_REFINE_MARGIN_WEIGHT"])
    module.CROP_REFINE_BLACK_WEIGHT = float(base["CROP_REFINE_BLACK_WEIGHT"])
    module.CROP_REFINE_EDGE_WEIGHT = float(base["CROP_REFINE_EDGE_WEIGHT"])
    module.EARTH_SECONDARY_THRESHOLD = float(base["EARTH_SECONDARY_THRESHOLD"])
    module.classify_shifting_earth = base["classify_shifting_earth"]


def install_earth_classifier(module, low_weight, grad_weight) -> None:
    if low_weight is None and grad_weight is None:
        return
    lw = 0.65 if low_weight is None else float(low_weight)
    gw = 0.35 if grad_weight is None else float(grad_weight)
    total = abs(lw) + abs(gw)
    if total <= 1e-8:
        lw, gw = 0.65, 0.35
    else:
        lw, gw = lw / total, gw / total

    def classify_shifting_earth(panel_bgr, package):
        shot_low = module.preprocess_for_similarity(panel_bgr)
        shot_grad = module.preprocess_for_gradients(panel_bgr)
        shot_low_n = module.normalize(shot_low)
        shot_grad_n = module.normalize(shot_grad)
        ranked = []
        for idx, earth_name in enumerate(package.earth_names):
            low_corr = module.corr(shot_low_n, module.normalize(package.terrain_low[idx]))
            grad_corr = module.corr(shot_grad_n, module.normalize(package.terrain_grad[idx]))
            score = float(lw) * low_corr + float(gw) * grad_corr
            ranked.append((earth_name, score, {"low_corr": low_corr, "grad_corr": grad_corr}))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    module.classify_shifting_earth = classify_shifting_earth


def apply_config_to_module(module, cfg: dict) -> None:
    base = capture_base_state(module)
    restore_base_state(module, base)

    if "template_scales" in cfg and cfg["template_scales"]:
        module.DEFAULT_TEMPLATE_SCALES = tuple(float(x) for x in cfg["template_scales"])

    if "crop_refine_margin_weight" in cfg:
        module.CROP_REFINE_MARGIN_WEIGHT = float(cfg["crop_refine_margin_weight"])
    if "crop_refine_black_weight" in cfg:
        module.CROP_REFINE_BLACK_WEIGHT = float(cfg["crop_refine_black_weight"])
    if "crop_refine_edge_weight" in cfg:
        module.CROP_REFINE_EDGE_WEIGHT = float(cfg["crop_refine_edge_weight"])
    if "earth_secondary_threshold" in cfg:
        module.EARTH_SECONDARY_THRESHOLD = float(cfg["earth_secondary_threshold"])

    type_overrides = cfg.get("type_overrides", {})
    if isinstance(type_overrides, dict):
        for type_name, overrides in type_overrides.items():
            if type_name not in module.TYPE_CONFIG or not isinstance(overrides, dict):
                continue
            for key, value in overrides.items():
                module.TYPE_CONFIG[type_name][key] = float(value)

    install_earth_classifier(
        module,
        cfg.get("earth_low_weight"),
        cfg.get("earth_grad_weight"),
    )


def load_model_bundle(model_path: Path, device: torch.device):
    bundle = torch.load(model_path, map_location=device)
    labels = bundle["labels"]
    model = SmallSlotCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    patch_size = int(bundle.get("patch_size", 96))
    context_scale = float(bundle.get("context_scale", 1.8))
    min_half_frac = float(bundle.get("min_half_frac", 0.075))
    return model, labels, patch_size, context_scale, min_half_frac


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


def patch_to_tensor_rgb(patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0


def infer_evergaol_present_prob(bundle: dict, patch_bgr: np.ndarray) -> float:
    x = patch_to_tensor_rgb(patch_bgr, bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    labels = list(bundle["labels"])
    present_idx = labels.index("present") if "present" in labels else min(1, len(labels) - 1)
    return float(probs[present_idx])


def patch_to_tensor(patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return x.to(device)


@torch.no_grad()
def infer_slot_probs(model: nn.Module, labels: Sequence[str], patch_bgr: np.ndarray, device: torch.device) -> Dict[str, float]:
    x = patch_to_tensor(patch_bgr, device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {label: float(prob) for label, prob in zip(labels, probs)}


def apply_slot_label_constraints(probs: Dict[str, float], slot: dict) -> Tuple[Dict[str, float], List[str]]:
    allowed = slot.get("allowed_labels")
    if not isinstance(allowed, list) or not allowed:
        allowed_types = slot.get("allowed_types")
        if isinstance(allowed_types, list) and allowed_types:
            allowed = ["empty"] + [str(name) for name in allowed_types if str(name) != "empty"]
        else:
            allowed = list(probs.keys())

    allowed_labels = []
    seen = set()
    for name in allowed:
        label = str(name)
        if label in probs and label not in seen:
            allowed_labels.append(label)
            seen.add(label)
    if not allowed_labels:
        allowed_labels = list(probs.keys())

    constrained = {label: (float(prob) if label in seen else 0.0) for label, prob in probs.items()}
    total = float(sum(constrained.values()))
    if total > 1e-12:
        constrained = {label: (value / total) for label, value in constrained.items()}
    return constrained, allowed_labels


def ai_classify_slots(
    panel_bgr: np.ndarray,
    backend,
    package,
    earth_name: str,
    model: nn.Module,
    labels: Sequence[str],
    patch_size: int,
    context_scale: float,
    min_half_frac: float,
    device: torch.device,
    confidence_threshold: float,
    margin_threshold: float,
):
    dx_norm, dy_norm, align_response = estimate_global_translation_compat(backend, panel_bgr, package, earth_name)
    h, w = panel_bgr.shape[:2]
    side = min(h, w)
    slots_out = []

    for slot in package.slots_by_earth[earth_name]:
        cx = int(round((float(slot["x_norm"]) + dx_norm) * w))
        cy = int(round((float(slot["y_norm"]) + dy_norm) * h))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        half = int(round(max(float(slot["radius_norm"]) * side * context_scale, min_half_frac * side)))
        half = max(18, half)

        patch = reflect_pad_extract(panel_bgr, cx, cy, half, patch_size)
        raw_probs = infer_slot_probs(model, labels, patch, device)
        probs, allowed_labels = apply_slot_label_constraints(raw_probs, slot)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_prob = ordered[0]
        second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_prob - second_prob

        if best_prob < confidence_threshold or margin < margin_threshold:
            final_label = "unknown"
        else:
            final_label = best_label

        slots_out.append({
            "slot_id": int(slot["slot_id"]),
            "slot_name": str(slot["slot_name"]),
            "label": final_label,
            "best_prob": float(best_prob),
            "second_prob": float(second_prob),
            "margin": float(margin),
            "probs": probs,
            "raw_probs": raw_probs,
            "allowed_labels": allowed_labels,
            "x_norm": float(cx / w),
            "y_norm": float(cy / h),
            "nominal_x_norm": float(slot["x_norm"]),
            "nominal_y_norm": float(slot["y_norm"]),
            "radius_norm": float(slot["radius_norm"]),
            "weight": float(slot.get("weight", 1.0)),
        })

    return slots_out, {
        "dx_norm": float(dx_norm),
        "dy_norm": float(dy_norm),
        "align_response": float(align_response),
    }


SPECIAL_LAYOUT_COMPARISONS: Tuple[Tuple[int, int], ...] = (
    (101, 1089),
)
GREAT_HOLLOW_ICON_ONLY_SLOT_IDS = {2}


def score_layout_ai(
    slots_out: Sequence[dict],
    layout_slot_labels: Sequence[str],
    earth_score: float,
    earth_name: str = "",
    geometry_prior: float = 0.0,
    lost_slot_prior: float = 0.0,
    hard_constraint_prior: float = 0.0,
    evergaol_prior: float = 0.0,
    great_hollow_status_prior: float = 0.0,
) -> Tuple[float, dict]:
    eps = 1e-6
    total = 0.0
    total_weight = 0.0
    exact_weight = 0.0
    unknown_weight = 0.0
    hard_mismatch_weight = 0.0
    slot_details = []
    for slot_obs, expected in zip(slots_out, layout_slot_labels):
        slot_id = int(slot_obs.get("slot_id", -1))
        if str(earth_name) == "Great Hollow" and slot_id in GREAT_HOLLOW_ICON_ONLY_SLOT_IDS:
            slot_details.append({
                "slot_id": int(slot_id),
                "slot_name": str(slot_obs["slot_name"]),
                "expected": str(expected),
                "observed": str(slot_obs.get("label", "unknown")),
                "expected_prob": None,
                "best_prob": float(slot_obs.get("best_prob", 0.0)),
                "second_prob": float(slot_obs.get("second_prob", 0.0)),
                "margin": float(slot_obs.get("margin", 0.0)),
                "weight": 0.0,
                "weighted_log_prob": 0.0,
                "icon_only_slot": True,
            })
            continue
        probs = slot_obs["probs"]
        p = max(eps, float(probs.get(str(expected), eps)))
        weight = float(slot_obs.get("weight", 1.0))
        total += weight * math.log(p)
        total_weight += weight
        observed = str(slot_obs["label"])
        if observed == str(expected):
            exact_weight += weight
        elif observed == "unknown":
            unknown_weight += weight
        else:
            hard_mismatch_weight += weight
        slot_details.append({
            "slot_id": int(slot_id),
            "slot_name": str(slot_obs["slot_name"]),
            "expected": str(expected),
            "observed": observed,
            "expected_prob": float(p),
            "best_prob": float(slot_obs.get("best_prob", 0.0)),
            "second_prob": float(slot_obs.get("second_prob", 0.0)),
            "margin": float(slot_obs.get("margin", 0.0)),
            "weight": float(weight),
            "weighted_log_prob": float(weight * math.log(p)),
            "icon_only_slot": False,
        })
    normalized = total / max(1.0, total_weight)
    total_score = (
        normalized
        + 0.75 * float(earth_score)
        + float(geometry_prior)
        + float(lost_slot_prior)
        + float(hard_constraint_prior)
        + float(evergaol_prior)
        + float(great_hollow_status_prior)
    )
    details = {
        "normalized_log_prob": float(normalized),
        "geometry_prior": float(geometry_prior),
        "lost_slot_prior": float(lost_slot_prior),
        "hard_constraint_prior": float(hard_constraint_prior),
        "evergaol_prior": float(evergaol_prior),
        "great_hollow_status_prior": float(great_hollow_status_prior),
        "exact_match_weight": float(exact_weight),
        "hard_mismatch_weight": float(hard_mismatch_weight),
        "unknown_weight": float(unknown_weight),
        "slot_details": slot_details,
    }
    return float(total_score), details


def compute_earth_geometry_priors(package, observations_by_earth: dict) -> dict:
    slot_counts = {earth_name: len(package.slots_by_earth.get(earth_name, [])) for earth_name in observations_by_earth}
    standard = {
        earth_name: float(obs["align"]["align_response"])
        for earth_name, obs in observations_by_earth.items()
        if slot_counts.get(earth_name, 0) >= 20
    }
    compact = {
        earth_name: float(obs["align"]["align_response"])
        for earth_name, obs in observations_by_earth.items()
        if slot_counts.get(earth_name, 0) <= 10
    }
    best_standard = max(standard.values()) if standard else 0.0
    best_compact = max(compact.values()) if compact else 0.0
    priors = {}
    for earth_name, obs in observations_by_earth.items():
        align_response = float(obs["align"]["align_response"])
        slot_count = slot_counts.get(earth_name, 0)
        prior = 0.0
        if slot_count <= 10 and standard:
            gap = best_standard - align_response
            if gap > 0.0:
                prior -= 2.0 * gap
            if gap > 0.010:
                prior -= 0.08
            if gap > 0.020:
                prior -= 0.08
        elif slot_count >= 20 and compact:
            gap = align_response - best_compact
            if gap > 0.0:
                prior += 0.75 * gap
        priors[earth_name] = {
            "slot_count": int(slot_count),
            "align_response": float(align_response),
            "best_standard_align": float(best_standard),
            "best_compact_align": float(best_compact),
            "geometry_prior": float(prior),
        }
    return priors


def derive_lost_limveld_slot_indices(package, match_radius_norm: float = 0.035) -> dict:
    limveld_slots = list(package.slots_by_earth.get("Limveld", []))
    if not limveld_slots:
        return {}
    out = {}
    for earth_name, slots in package.slots_by_earth.items():
        if earth_name in {"Limveld", "Great Hollow"}:
            continue
        matched_limveld = set()
        for slot in slots:
            sx = float(slot["x_norm"])
            sy = float(slot["y_norm"])
            best_idx = None
            best_dist = float("inf")
            for idx, limveld_slot in enumerate(limveld_slots):
                dx = sx - float(limveld_slot["x_norm"])
                dy = sy - float(limveld_slot["y_norm"])
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist <= match_radius_norm:
                matched_limveld.add(int(best_idx))
        out[earth_name] = [idx for idx in range(len(limveld_slots)) if idx not in matched_limveld]
    return out


def compute_lost_limveld_slot_priors(package, observations_by_earth: dict) -> dict:
    limveld_obs = observations_by_earth.get("Limveld", {}).get("slots")
    if not limveld_obs:
        return {}

    lost_indices_by_earth = derive_lost_limveld_slot_indices(package)
    priors = {}
    for earth_name in observations_by_earth:
        lost_indices = list(lost_indices_by_earth.get(earth_name, []))
        prior = 0.0
        slot_evidence = []
        for idx in lost_indices:
            if idx < 0 or idx >= len(limveld_obs):
                continue
            obs = limveld_obs[idx]
            probs = dict(obs.get("probs", {}))
            empty_prob = float(probs.get("empty", 0.0))
            nonempty_prob = max(0.0, 1.0 - empty_prob)
            label = str(obs.get("label", "unknown"))
            slot_prior = 0.0
            if label == "empty":
                empty_strength = max(0.0, empty_prob - 0.55) / 0.45
                slot_prior += 0.010 + 0.015 * min(1.0, empty_strength)
            elif label == "unknown":
                if nonempty_prob > 0.70:
                    slot_prior -= 0.025 * min(1.0, (nonempty_prob - 0.70) / 0.30)
            else:
                occupied_strength = max(0.0, nonempty_prob - 0.45) / 0.55
                slot_prior -= 0.050 + 0.050 * min(1.0, occupied_strength)
            prior += slot_prior
            slot_evidence.append({
                "slot_id": int(obs.get("slot_id", idx + 1)),
                "slot_name": str(obs.get("slot_name", f"slot_{idx+1:02d}")),
                "label": label,
                "empty_prob": float(empty_prob),
                "nonempty_prob": float(nonempty_prob),
                "slot_prior": float(slot_prior),
            })
        priors[earth_name] = {
            "lost_slot_count": int(len(lost_indices)),
            "lost_slot_prior": float(prior),
            "slot_evidence": slot_evidence,
        }
    return priors


def compute_hard_constraint_priors(package, observations_by_earth: dict) -> dict:
    priors = {
        earth_name: {
            "hard_constraint_prior": 0.0,
            "evidence": [],
        }
        for earth_name in observations_by_earth
    }
    limveld_obs = observations_by_earth.get("Limveld", {}).get("slots")
    if not limveld_obs:
        return priors

    matched_to_great_hollow = set(derive_lost_limveld_slot_indices(package).get("Great Hollow", []))
    # `derive_lost_limveld_slot_indices` excludes Great Hollow by design. Rebuild the complement here.
    limveld_slots = list(package.slots_by_earth.get("Limveld", []))
    great_hollow_slots = list(package.slots_by_earth.get("Great Hollow", []))
    if limveld_slots and great_hollow_slots:
        matched_limveld = set()
        for gh_slot in great_hollow_slots:
            sx = float(gh_slot["x_norm"])
            sy = float(gh_slot["y_norm"])
            best_idx = None
            best_dist = float("inf")
            for idx, limveld_slot in enumerate(limveld_slots):
                dx = sx - float(limveld_slot["x_norm"])
                dy = sy - float(limveld_slot["y_norm"])
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist <= 0.050:
                matched_limveld.add(int(best_idx))
        forbidden_limveld = [idx for idx in range(len(limveld_slots)) if idx not in matched_limveld]
    else:
        forbidden_limveld = []

    evidence = []
    prior = 0.0
    ranked = []
    for idx in forbidden_limveld:
        if idx < 0 or idx >= len(limveld_obs):
            continue
        obs = limveld_obs[idx]
        probs = dict(obs.get("probs", {}))
        empty_prob = float(probs.get("empty", 0.0))
        nonempty_prob = max(0.0, 1.0 - empty_prob)
        ranked.append((nonempty_prob, idx, obs))
    ranked.sort(reverse=True, key=lambda item: item[0])
    for nonempty_prob, idx, obs in ranked:
        label = str(obs.get("label", "unknown"))
        slot_prior = 0.0
        if label == "unknown":
            if nonempty_prob > 0.75:
                slot_prior -= 0.035 * min(1.0, (nonempty_prob - 0.75) / 0.25)
        elif label != "empty":
            occupied_strength = max(0.0, nonempty_prob - 0.40) / 0.60
            slot_prior -= 0.075 + 0.055 * min(1.0, occupied_strength)
        prior += slot_prior
        evidence.append({
            "slot_id": int(obs.get("slot_id", idx + 1)),
            "slot_name": str(obs.get("slot_name", f"slot_{idx+1:02d}")),
            "label": label,
            "nonempty_prob": float(nonempty_prob),
            "slot_prior": float(slot_prior),
        })
    if "Great Hollow" in priors:
        priors["Great Hollow"] = {
            "hard_constraint_prior": float(prior),
            "evidence": evidence,
        }
    return priors


def world_to_panel_norm(x_world: float, z_world: float) -> Tuple[float, float]:
    px = (float(x_world) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)
    py = 1.0 - ((float(z_world) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z))
    return float(px), float(py)


def load_evergaol_templates() -> List[dict]:
    global EVERGAOL_TEMPLATE_CACHE
    if EVERGAOL_TEMPLATE_CACHE is not None:
        return EVERGAOL_TEMPLATE_CACHE
    template = cv2.imread(str(EVERGAOL_ICON_PATH), cv2.IMREAD_UNCHANGED)
    if template is None:
        EVERGAOL_TEMPLATE_CACHE = []
        return EVERGAOL_TEMPLATE_CACHE
    if template.ndim == 2:
        gray = template
        alpha = np.full_like(gray, 255)
    elif template.shape[2] == 4:
        bgr = template[:, :, :3]
        alpha = template[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = template[:, :, :3]
        alpha = np.full(template.shape[:2], 255, dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    templates: List[dict] = []
    for scale in (0.55, 0.65, 0.75, 0.85, 0.95):
        w = max(16, int(round(gray.shape[1] * scale)))
        h = max(16, int(round(gray.shape[0] * scale)))
        gray_rs = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        edge_rs = cv2.resize(edge, (w, h), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        templates.append({
            "scale": float(scale),
            "gray": gray_rs,
            "edge": edge_rs,
            "mask": mask_rs,
            "w": int(w),
            "h": int(h),
        })
    EVERGAOL_TEMPLATE_CACHE = templates
    return templates


def detect_evergaol_presence(panel_bgr: np.ndarray, dx_norm: float = 0.0, dy_norm: float = 0.0, device: Optional[torch.device] = None) -> List[dict]:
    templates = load_evergaol_templates()
    evergaol_bundle = load_evergaol_model_bundle(device or choose_device("auto"))
    if not templates:
        return []
    panel_gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    panel_edge = cv2.Canny(panel_gray, 60, 180)
    h, w = panel_gray.shape[:2]
    side = min(h, w)
    search_radius = max(28, int(round(side * 0.045)))
    detections = []
    for idx, (x_world, z_world) in enumerate(EVERGAOL_WORLD_POINTS):
        x_norm, y_norm = world_to_panel_norm(x_world, z_world)
        cx = int(round((x_norm + dx_norm) * w))
        cy = int(round((y_norm + dy_norm) * h))
        best = {
            "score": -1.0,
            "gray_score": -1.0,
            "edge_score": -1.0,
            "scale": None,
            "x_norm": float(cx / max(1, w)),
            "y_norm": float(cy / max(1, h)),
            "present_prob": 0.0,
            "method": "template",
        }
        ai_best = None
        if evergaol_bundle is not None:
            patch_size = int(evergaol_bundle["patch_size"])
            context_scale = float(evergaol_bundle["context_scale"])
            min_half_frac = float(evergaol_bundle["min_half_frac"])
            offsets = (-search_radius, -search_radius // 2, 0, search_radius // 2, search_radius)
            for ox in offsets:
                for oy in offsets:
                    px = max(0, min(w - 1, cx + int(ox)))
                    py = max(0, min(h - 1, cy + int(oy)))
                    half = int(round(max(side * context_scale * 0.018, min_half_frac * side)))
                    half = max(20, half)
                    patch = reflect_pad_extract(panel_bgr, px, py, half, patch_size)
                    present_prob = infer_evergaol_present_prob(evergaol_bundle, patch)
                    if ai_best is None or present_prob > float(ai_best["present_prob"]):
                        ai_best = {
                            "score": float(present_prob),
                            "gray_score": float("nan"),
                            "edge_score": float("nan"),
                            "scale": None,
                            "x_norm": float(px / w),
                            "y_norm": float(py / h),
                            "present_prob": float(present_prob),
                            "method": "ai",
                        }
        template_best = dict(best)
        for tpl in templates:
            tw = int(tpl["w"])
            th = int(tpl["h"])
            x0 = max(0, cx - search_radius - tw // 2)
            y0 = max(0, cy - search_radius - th // 2)
            x1 = min(w, cx + search_radius + (tw + 1) // 2)
            y1 = min(h, cy + search_radius + (th + 1) // 2)
            roi_gray = panel_gray[y0:y1, x0:x1]
            roi_edge = panel_edge[y0:y1, x0:x1]
            if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
                continue
            gray_score_map = cv2.matchTemplate(roi_gray, tpl["gray"], cv2.TM_CCORR_NORMED, mask=tpl["mask"])
            edge_score_map = cv2.matchTemplate(roi_edge, tpl["edge"], cv2.TM_CCORR_NORMED, mask=tpl["mask"])
            gray_score = float(np.nanmax(gray_score_map))
            edge_score = float(np.nanmax(edge_score_map))
            score = 0.65 * gray_score + 0.35 * edge_score
            if score > float(template_best["score"]):
                _, _, _, max_loc = cv2.minMaxLoc(gray_score_map)
                px = x0 + max_loc[0] + tw * 0.5
                py = y0 + max_loc[1] + th * 0.5
                template_best = {
                    "score": float(score),
                    "gray_score": float(gray_score),
                    "edge_score": float(edge_score),
                    "scale": float(tpl["scale"]),
                    "x_norm": float(px / w),
                    "y_norm": float(py / h),
                    "present_prob": min(1.0, max(0.0, (float(score) - 0.30) / 0.45)),
                    "method": "template",
                }
        if ai_best is not None:
            hybrid_prob = 0.35 * float(ai_best["present_prob"]) + 0.65 * float(template_best["present_prob"])
            best = dict(template_best)
            best["present_prob"] = float(hybrid_prob)
            best["method"] = "hybrid"
            best["ai_present_prob"] = float(ai_best["present_prob"])
            best["template_present_prob"] = float(template_best["present_prob"])
        else:
            best = template_best
        present_prob = float(best["present_prob"])
        detections.append({
            "evergaol_id": int(idx + 1),
            "world_x": float(x_world),
            "world_z": float(z_world),
            "expected_x_norm": float(x_norm + dx_norm),
            "expected_y_norm": float(y_norm + dy_norm),
            **best,
        })
    return detections


def compute_evergaol_priors(panel_bgr: np.ndarray, observations_by_earth: dict, device: Optional[torch.device] = None) -> dict:
    align = observations_by_earth.get("Limveld", {}).get("align", {})
    dx_norm = float(align.get("dx_norm", 0.0))
    dy_norm = float(align.get("dy_norm", 0.0))
    detections = detect_evergaol_presence(panel_bgr, dx_norm=dx_norm, dy_norm=dy_norm, device=device)
    priors = {}
    for earth_name in observations_by_earth:
        expected = set(EVERGAOL_EXPECTED_BY_EARTH.get(earth_name, set()))
        prior = 0.0
        evidence = []
        for idx, det in enumerate(detections):
            p = float(det["present_prob"])
            if idx in expected:
                slot_prior = 0.09 * (p - 0.35)
            else:
                slot_prior = 0.10 * (0.35 - p)
            prior += slot_prior
            evidence.append({
                "evergaol_id": int(det["evergaol_id"]),
                "expected_present": bool(idx in expected),
                "present_prob": float(p),
                "match_score": float(det["score"]),
                "slot_prior": float(slot_prior),
            })
        priors[earth_name] = {
            "evergaol_prior": float(prior),
            "detections": detections,
            "evidence": evidence,
        }
    return priors


def draw_evergaol_debug(panel_bgr: np.ndarray, evergaol_priors: dict) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    limveld = evergaol_priors.get("Limveld", {})
    detections = list(limveld.get("detections", []))
    for idx, det in enumerate(detections):
        ex = int(round(float(det["expected_x_norm"]) * w))
        ey = int(round(float(det["expected_y_norm"]) * h))
        px = int(round(float(det["x_norm"]) * w))
        py = int(round(float(det["y_norm"]) * h))
        prob = float(det.get("present_prob", 0.0))
        color = (0, 255, 0) if prob >= 0.5 else (0, 128, 255)
        cv2.circle(out, (ex, ey), 14, (255, 255, 0), 1)
        cv2.circle(out, (px, py), 8, color, -1)
        text = f"E{idx+1}: {prob:.2f} {det.get('method','?')}"
        tx = max(6, min(w - 220, px + 8))
        ty = max(16, min(h - 8, py - 8))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _load_icon_gray_edge_mask(path: Path):
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 2:
        gray = icon
        alpha = np.full_like(gray, 255)
    elif icon.shape[2] == 4:
        bgr = icon[:, :, :3]
        alpha = icon[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = icon[:, :, :3]
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    return {"gray": gray, "edge": edge, "mask": mask}


def _load_icon_color_gray_edge_mask(path: Path):
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 2:
        bgr = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
    elif icon.shape[2] == 4:
        bgr = icon[:, :, :3]
        alpha = icon[:, :, 3]
    else:
        bgr = icon[:, :, :3]
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    return {"bgr": bgr, "gray": gray, "edge": edge, "mask": mask}


def load_great_hollow_status_icon_sources() -> dict:
    global GREAT_HOLLOW_STATUS_ICON_SOURCES
    if GREAT_HOLLOW_STATUS_ICON_SOURCES is not None:
        return GREAT_HOLLOW_STATUS_ICON_SOURCES
    out = {}
    for icon_name, path in GREAT_HOLLOW_STATUS_ICON_PATHS.items():
        data = _load_icon_color_gray_edge_mask(path)
        if data is not None:
            out[icon_name] = data
    GREAT_HOLLOW_STATUS_ICON_SOURCES = out
    return out


def _find_great_hollow_pattern_path(layout_number: int) -> Optional[Path]:
    for base_dir in GREAT_HOLLOW_STATUS_PATTERN_DIRS:
        matches = sorted(base_dir.glob(f"pattern_{int(layout_number):04d}_*.json"))
        if matches:
            return matches[0]
    return None


def _great_hollow_world_to_norm(x_world: float, z_world: float) -> Tuple[float, float]:
    px = (float(x_world) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = (float(z_world) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H
    cropped_w = IMAGE_W - (2.0 * GREAT_HOLLOW_CROP_BORDER)
    cropped_h = IMAGE_H - (2.0 * GREAT_HOLLOW_CROP_BORDER)
    x_norm = (px - GREAT_HOLLOW_CROP_BORDER) / cropped_w
    y_norm = (py - GREAT_HOLLOW_CROP_BORDER) / cropped_h
    return float(x_norm), float(y_norm)


def _load_great_hollow_patternset_statuses(layout_number: int) -> dict:
    pattern_path = GREAT_HOLLOW_PATTERNSET_DIR / f"pattern_{int(layout_number):04d}_mappatternset_bundle.json"
    if not pattern_path.exists():
        return {}
    try:
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    statuses = {}
    for item in payload.get("activeMapPatternSetFields", []) or []:
        field_name = str(item.get("field", "")).strip()
        mapping = GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD.get(field_name)
        if mapping is None:
            continue
        anchor_type, status_type = mapping
        point = GREAT_HOLLOW_SYNTHETIC_WORLD_POINTS.get(anchor_type)
        if point is None:
            continue
        x_norm, y_norm = _great_hollow_world_to_norm(point[0], point[1])
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "source_id": None,
        }
    return statuses


def get_layout_great_hollow_statuses(layout: dict) -> dict:
    layout_number = int(layout.get("layout_number", -1))
    cached = GREAT_HOLLOW_LAYOUT_STATUS_CACHE.get(layout_number)
    if cached is not None:
        return cached

    statuses = {}
    for item in layout.get("great_hollow_statuses", []) or []:
        anchor_type = str(item.get("anchor_type", "")).strip()
        status_type = str(item.get("status_type", "")).strip()
        if anchor_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
            continue
        if status_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE[anchor_type]:
            continue
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": float(item.get("x_norm", 0.0)),
            "y_norm": float(item.get("y_norm", 0.0)),
            "source_id": item.get("source_id", None),
        }
    if statuses:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = statuses
        return statuses

    pattern_path = _find_great_hollow_pattern_path(layout_number)
    if pattern_path is None:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = {}
        return {}
    try:
        pattern_data = json.loads(pattern_path.read_text(encoding="utf-8"))
    except Exception:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = {}
        return {}

    by_source_id = {}
    for item in pattern_data.get("landmarks", []):
        by_source_id.setdefault(item.get("source_id", None), []).append(item)

    for source_id, items in by_source_id.items():
        overlay_item = None
        base_item = None
        for item in items:
            item_type = str(item.get("type", "")).strip()
            if item_type in GREAT_HOLLOW_STATUS_ICON_PATHS:
                overlay_item = item
            if item.get("source") != "overlay" and item_type in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
                base_item = item
        if overlay_item is None or base_item is None:
            continue
        anchor_type = str(base_item.get("type", "")).strip()
        status_type = str(overlay_item.get("type", "")).strip()
        if status_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(anchor_type, ()):
            continue
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": float(overlay_item.get("x_norm", 0.0)),
            "y_norm": float(overlay_item.get("y_norm", 0.0)),
            "source_id": source_id,
        }
    bundle_statuses = _load_great_hollow_patternset_statuses(layout_number)
    for anchor_type, info in bundle_statuses.items():
        statuses.setdefault(anchor_type, info)
    GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = statuses
    return statuses


def get_great_hollow_status_anchor_norms(package) -> dict:
    global GREAT_HOLLOW_STATUS_ANCHOR_CACHE
    if GREAT_HOLLOW_STATUS_ANCHOR_CACHE is not None:
        return GREAT_HOLLOW_STATUS_ANCHOR_CACHE

    anchors = {}
    gh_layouts = list(package.layouts_by_earth.get("Great Hollow", []))
    for anchor_type in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
        xs = []
        ys = []
        for layout in gh_layouts:
            info = get_layout_great_hollow_statuses(layout).get(anchor_type)
            if info is None:
                continue
            xs.append(float(info["x_norm"]))
            ys.append(float(info["y_norm"]))
        if xs and ys:
            anchors[anchor_type] = {
                "x_norm": float(sum(xs) / len(xs)),
                "y_norm": float(sum(ys) / len(ys)),
            }
            continue
        for slot in package.slots_by_earth.get("Great Hollow", []):
            allowed = {str(name) for name in slot.get("allowed_types", [])}
            if allowed == {anchor_type}:
                offset_norm = 23.0 / 1280.0
                anchors[anchor_type] = {
                    "x_norm": float(slot["x_norm"]) + offset_norm,
                    "y_norm": float(slot["y_norm"]) + offset_norm,
                }
                break
    GREAT_HOLLOW_STATUS_ANCHOR_CACHE = anchors
    return anchors


def _gh_read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _gh_resize_to_match(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    if reference.shape[:2] == query.shape[:2]:
        return query
    return cv2.resize(query, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def _gh_crop_with_padding(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + size)
    y1 = min(h, y + size)
    patch = img[y0:y1, x0:x1].copy()
    if patch.size == 0:
        raise RuntimeError(f"Empty crop at x={x}, y={y}, size={size} for image size {w}x{h}.")
    top = y0 - y
    left = x0 - x
    bottom = (y + size) - y1
    right = (x + size) - x1
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return patch


def _gh_get_fixed_status_params(img: np.ndarray, xy_norm: Tuple[float, float]) -> Tuple[Tuple[int, int], int, int]:
    h, w = img.shape[:2]
    x = int(round(xy_norm[0] * w))
    y = int(round(xy_norm[1] * h))
    scale = min(h, w) / 1280.0
    crop_size = max(24, int(round(72 * scale)))
    search_pad = max(4, int(round(12 * scale)))
    return (x, y), crop_size, search_pad


def _gh_feature_vector(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hh, ww = h.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    cy, cx = hh / 2.0, ww / 2.0
    radial = np.exp(-((((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5) / (min(hh, ww) / 2.6)) ** 2)
    sat_mask = (s > 50) & (v > 40)
    weights = radial * sat_mask.astype(np.float32)
    if float(weights.sum()) < 1.0:
        weights = radial.astype(np.float32)

    angle = h.astype(np.float32) / 180.0 * 2.0 * np.pi
    hue_x = float((np.cos(angle) * weights).sum() / weights.sum())
    hue_y = float((np.sin(angle) * weights).sum() / weights.sum())
    mean_hue = float(math.atan2(hue_y, hue_x))
    mean_sat = float((s * weights).sum() / weights.sum() / 255.0)
    mean_val = float((v * weights).sum() / weights.sum() / 255.0)

    color_hist = cv2.calcHist([hsv], [0, 1], sat_mask.astype(np.uint8), [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
    if float(color_hist.sum()) > 0:
        color_hist /= color_hist.sum()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    grad_hist = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag * radial)[0].astype(np.float32)
    if float(grad_hist.sum()) > 0:
        grad_hist /= grad_hist.sum()

    return {
        "hue": mean_hue,
        "sat": mean_sat,
        "val": mean_val,
        "color_hist": color_hist,
        "grad_hist": grad_hist,
    }


def _gh_circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2 * np.pi - d) / np.pi


def _gh_patch_similarity(a: np.ndarray, b: np.ndarray) -> float:
    fa = _gh_feature_vector(a)
    fb = _gh_feature_vector(b)
    hue_score = 1.0 - _gh_circular_distance(float(fa["hue"]), float(fb["hue"]))
    hist_score = float(cv2.compareHist(np.asarray(fa["color_hist"], dtype=np.float32), np.asarray(fb["color_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["color_hist"]).sum() > 0 and np.asarray(fb["color_hist"]).sum() > 0 else 0.0
    grad_score = float(cv2.compareHist(np.asarray(fa["grad_hist"], dtype=np.float32), np.asarray(fb["grad_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["grad_hist"]).sum() > 0 and np.asarray(fb["grad_hist"]).sum() > 0 else 0.0
    sv_score = 1.0 - (abs(float(fa["sat"]) - float(fb["sat"])) + abs(float(fa["val"]) - float(fb["val"]))) / 2.0
    return float(0.45 * hue_score + 0.30 * hist_score + 0.15 * grad_score + 0.10 * sv_score)


def _gh_classify_fixed_status_window(search: np.ndarray, templates: dict) -> Tuple[List[Tuple[str, float, Tuple[int, int]]], Tuple[int, int]]:
    template_h, template_w = next(iter(templates.values())).shape[:2]
    results = []
    for label, template in templates.items():
        best_score = -1.0
        best_loc = (0, 0)
        for dy in range(0, search.shape[0] - template_h + 1):
            for dx in range(0, search.shape[1] - template_w + 1):
                patch = search[dy:dy + template_h, dx:dx + template_w]
                score = _gh_patch_similarity(patch, template)
                if score > best_score:
                    best_score = score
                    best_loc = (dx, dy)
        results.append((label, float(best_score), best_loc))
    results.sort(key=lambda row: row[1], reverse=True)
    return results, (template_w, template_h)


def load_great_hollow_fixed_status_templates() -> dict:
    global GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE
    if GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE is not None:
        return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE

    ref1 = None
    ref2 = None
    for path in GREAT_HOLLOW_FIXED_STATUS_REFS["ref1"]:
        ref1 = _gh_read_image(path)
        if ref1 is not None:
            break
    for path in GREAT_HOLLOW_FIXED_STATUS_REFS["ref2"]:
        ref2 = _gh_read_image(path)
        if ref2 is not None:
            break
    if ref1 is None or ref2 is None:
        GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = {}
        return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE

    ref2 = _gh_resize_to_match(ref1, ref2)
    templates = {}
    for anchor_type, cfg in GREAT_HOLLOW_FIXED_STATUS_CONFIG.items():
        (x, y), crop_size, search_pad = _gh_get_fixed_status_params(ref1, cfg["xy_norm"])
        templates[anchor_type] = {
            "templates": {
                str(cfg["ref1_label"]): _gh_crop_with_padding(ref1, x, y, crop_size),
                str(cfg["ref2_label"]): _gh_crop_with_padding(ref2, x, y, crop_size),
            },
            "meta": {
                "xy_norm": tuple(float(v) for v in cfg["xy_norm"]),
                "crop_size": int(crop_size),
                "search_pad": int(search_pad),
            },
        }
    GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = templates
    return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE


def _load_icon_recognizer_v4_module():
    global ICON_RECOGNIZER_V4_MODULE, ICON_RECOGNIZER_V4_LOAD_ATTEMPTED
    if ICON_RECOGNIZER_V4_LOAD_ATTEMPTED:
        return ICON_RECOGNIZER_V4_MODULE
    ICON_RECOGNIZER_V4_LOAD_ATTEMPTED = True
    candidates = [
        Path("icon_recognizer_v4.py"),
        Path(__file__).resolve().parent / "icon_recognizer_v4.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("icon_recognizer_v4_runtime", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "recognize_markers"):
                ICON_RECOGNIZER_V4_MODULE = module
                return ICON_RECOGNIZER_V4_MODULE
        except Exception:
            continue
    ICON_RECOGNIZER_V4_MODULE = None
    return None


def _resolve_first_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
        resolved = Path(__file__).resolve().parent / path
        if resolved.exists():
            return resolved
    return None


def preprocess_panel_for_status_recognizer(panel_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        panel_bgr,
        (STATUS_RECOGNIZER_INPUT_SIZE, STATUS_RECOGNIZER_INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    b = int(STATUS_RECOGNIZER_BORDER_CROP_PX)
    return resized[b: STATUS_RECOGNIZER_INPUT_SIZE - b, b: STATUS_RECOGNIZER_INPUT_SIZE - b].copy()


def detect_great_hollow_statuses(panel_bgr: np.ndarray, package, observations_by_earth: dict) -> dict:
    recognizer = _load_icon_recognizer_v4_module()
    if recognizer is None:
        return {}

    ref1 = _resolve_first_existing_path(GREAT_HOLLOW_FIXED_STATUS_REFS["ref1"])
    ref2 = _resolve_first_existing_path(GREAT_HOLLOW_FIXED_STATUS_REFS["ref2"])
    if ref1 is None or ref2 is None:
        return {}

    label_map = {
        "Fire": "Status_Fire",
        "Magic": "Status_Magic",
        "Rot": "Status_Rot",
        "Holy": "Status_Holy",
        "Poison": "Status_Poison",
        "Bleed": "Status_Bleed",
        "Lightning": "Status_Lightning",
        "Sleep": "Status_Sleep",
    }
    position_map = {
        "Palace": "top_left",
        "Fortress": "bottom_right",
        "Ruins": "ruins",
        "Great Church": "great_church_top_right",
    }
    confidence_to_present_prob = {
        "strong": 0.95,
        "medium": 0.80,
        "weak": 0.60,
    }

    tmp_query_path = None
    tmp_ref1_path = None
    tmp_ref2_path = None
    status_panel = preprocess_panel_for_status_recognizer(panel_bgr)
    h, w = status_panel.shape[:2]
    try:
        ref1_img = cv2.imread(str(ref1), cv2.IMREAD_COLOR)
        ref2_img = cv2.imread(str(ref2), cv2.IMREAD_COLOR)
        if ref1_img is None or ref2_img is None:
            return {}

        ref1_status = preprocess_panel_for_status_recognizer(ref1_img)
        ref2_status = preprocess_panel_for_status_recognizer(ref2_img)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_q:
            tmp_query_path = Path(tmp_q.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_r1:
            tmp_ref1_path = Path(tmp_r1.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_r2:
            tmp_ref2_path = Path(tmp_r2.name)

        cv2.imwrite(str(tmp_query_path), status_panel)
        cv2.imwrite(str(tmp_ref1_path), ref1_status)
        cv2.imwrite(str(tmp_ref2_path), ref2_status)
        raw_result = recognizer.recognize_markers(str(tmp_ref1_path), str(tmp_ref2_path), str(tmp_query_path))
    except Exception:
        return {}
    finally:
        for tmp_path in (tmp_query_path, tmp_ref1_path, tmp_ref2_path):
            if tmp_path is None:
                continue
            try:
                tmp_path.unlink()
            except Exception:
                pass

    observed = {}
    for anchor_type, pos_key in position_map.items():
        slot = raw_result.get(pos_key, {})
        best_status = label_map.get(str(slot.get("label", "")))
        if not best_status:
            continue
        second_status = label_map.get(str(slot.get("runner_up", "")))
        best_score = float(slot.get("score", 0.0))
        second_score = float(slot.get("runner_up_score", 0.0))
        margin = float(slot.get("margin", best_score - second_score))
        conf_band = str(slot.get("confidence", "weak")).lower()
        present_prob = float(confidence_to_present_prob.get(conf_band, 0.60))
        box = slot.get("box", {})
        bx = float(box.get("x", 0.0))
        by = float(box.get("y", 0.0))
        bw = float(box.get("w", 0.0))
        bh = float(box.get("h", 0.0))
        cfg = GREAT_HOLLOW_FIXED_STATUS_CONFIG.get(anchor_type, {})
        xy_norm = cfg.get("xy_norm", (0.0, 0.0))

        ranked = []
        for item in slot.get("ranked", []):
            mapped = label_map.get(str(item.get("label", "")))
            if not mapped:
                continue
            ranked.append({
                "status_type": str(mapped),
                "score": float(item.get("score", 0.0)),
                "dx": int(item.get("dx", 0)),
                "dy": int(item.get("dy", 0)),
            })

        observed[anchor_type] = {
            "anchor_type": anchor_type,
            "expected_x_norm": float(xy_norm[0]),
            "expected_y_norm": float(xy_norm[1]),
            "best_status": str(best_status),
            "best_score": float(best_score),
            "second_status": str(second_status) if second_status else None,
            "second_score": float(second_score),
            "margin": float(margin),
            "present_prob": float(present_prob),
            "x_norm": float((bx + 0.5 * bw) / max(1, w)) if bw > 0 else float(xy_norm[0]),
            "y_norm": float((by + 0.5 * bh) / max(1, h)) if bh > 0 else float(xy_norm[1]),
            "method": "icon_recognizer_v4",
            "ranked": ranked,
        }
    return observed


def compute_great_hollow_status_prior(layout: dict, observed_statuses: dict) -> Tuple[float, dict]:
    if str(layout.get("shifting_earth", "")) != "Great Hollow":
        return 0.0, {"great_hollow_status_prior": 0.0, "status_evidence": []}
    expected_statuses = get_layout_great_hollow_statuses(layout)
    if not expected_statuses:
        return 0.0, {"great_hollow_status_prior": 0.0, "status_evidence": []}

    prior = 0.0
    evidence = []
    for anchor_type, expected in expected_statuses.items():
        observed = observed_statuses.get(anchor_type)
        if observed is None:
            continue
        allowed_statuses = tuple(str(s) for s in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(anchor_type, ()))
        best_score = float(observed.get("best_score", 0.0))
        margin = float(observed.get("margin", 0.0))
        score_strength = max(0.0, min(1.0, (best_score - 0.42) / 0.28))
        margin_strength = max(0.0, min(1.0, margin / 0.12))
        confidence = 0.55 * score_strength + 0.45 * margin_strength
        expected_status = str(expected.get("status_type", ""))
        observed_status = str(observed.get("best_status", ""))
        if observed_status == expected_status:
            slot_prior = 0.035 + 0.085 * confidence
        elif observed_status in allowed_statuses:
            slot_prior = -(0.030 + 0.110 * confidence)
        else:
            slot_prior = -0.180 - 0.100 * confidence
        prior += slot_prior
        evidence.append({
            "anchor_type": anchor_type,
            "allowed_statuses": list(allowed_statuses),
            "expected_status": expected_status,
            "observed_status": observed_status,
            "best_score": best_score,
            "margin": margin,
            "confidence": float(confidence),
            "slot_prior": float(slot_prior),
        })
    return float(prior), {
        "great_hollow_status_prior": float(prior),
        "status_evidence": evidence,
    }


def compute_great_hollow_family_support(observed_statuses: dict) -> dict:
    if not observed_statuses:
        return {"support": 0.0, "anchor_count": 0, "anchors": []}

    anchors = []
    total = 0.0
    for anchor_type, info in observed_statuses.items():
        best_score = float(info.get("best_score", 0.0))
        margin = float(info.get("margin", 0.0))
        present_prob = float(info.get("present_prob", 0.0))
        score_strength = max(0.0, min(1.0, (best_score - 0.42) / 0.30))
        margin_strength = max(0.0, min(1.0, margin / 0.08))
        anchor_support = 0.45 * present_prob + 0.35 * margin_strength + 0.20 * score_strength
        total += anchor_support
        anchors.append({
            "anchor_type": str(anchor_type),
            "best_status": str(info.get("best_status", "")),
            "best_score": float(best_score),
            "margin": float(margin),
            "present_prob": float(present_prob),
            "anchor_support": float(anchor_support),
        })

    return {
        "support": float(total / max(1, len(anchors))),
        "anchor_count": int(len(anchors)),
        "anchors": anchors,
    }


def rank_shifting_earths_cascade(
    earth_rank: Sequence[Tuple[str, float, dict]],
    geometry_priors: dict,
    lost_slot_priors: dict,
    hard_constraint_priors: dict,
    evergaol_priors: dict,
    great_hollow_status_observations: dict,
) -> List[Tuple[str, float, dict]]:
    base_by_earth = {str(name): (float(score), dict(details)) for name, score, details in earth_rank}
    gh_support = compute_great_hollow_family_support(great_hollow_status_observations)

    ranked = []
    for earth_name, (base_score, base_details) in base_by_earth.items():
        geometry_prior = float(geometry_priors.get(earth_name, {}).get("geometry_prior", 0.0))
        lost_slot_prior = float(lost_slot_priors.get(earth_name, {}).get("lost_slot_prior", 0.0))
        hard_constraint_prior = float(hard_constraint_priors.get(earth_name, {}).get("hard_constraint_prior", 0.0))
        evergaol_prior = float(evergaol_priors.get(earth_name, {}).get("evergaol_prior", 0.0))
        gh_status_support = float(gh_support.get("support", 0.0)) if earth_name == "Great Hollow" else 0.0

        if earth_name == "Great Hollow":
            cascade_score = (
                1.35 * base_score
                + 1.10 * geometry_prior
                + 1.45 * hard_constraint_prior
                + 1.15 * evergaol_prior
                + 0.10 * gh_status_support
            )
        else:
            structural_score = geometry_prior + 1.35 * lost_slot_prior + 1.35 * evergaol_prior
            cascade_score = 1.25 * base_score + structural_score

        ranked.append((
            earth_name,
            float(cascade_score),
            {
                "base_score": float(base_score),
                "geometry_prior": float(geometry_prior),
                "lost_slot_prior": float(lost_slot_prior),
                "hard_constraint_prior": float(hard_constraint_prior),
                "evergaol_prior": float(evergaol_prior),
                "great_hollow_status_support": float(gh_status_support),
                "raw_details": base_details,
            },
        ))

    by_earth = {name: {"score": score, "details": details} for name, score, details in ranked}
    gh_info = by_earth.get("Great Hollow")
    if gh_info is not None:
        gh_base = float(gh_info["details"].get("base_score", 0.0))
        best_non_gh_base = max(
            (float(item["details"].get("base_score", 0.0)) for name, item in by_earth.items() if name != "Great Hollow"),
            default=float("-inf"),
        )
        best_non_gh = max((item["score"] for name, item in by_earth.items() if name != "Great Hollow"), default=float("-inf"))
        gh_raw_dominance = gh_base >= 0.80 and gh_base >= best_non_gh_base + 0.25
        gh_gate_structural = (
            float(gh_info["score"]) >= float(best_non_gh) + 0.015
            and float(gh_info["details"].get("hard_constraint_prior", 0.0)) >= -0.18
        )
        gh_gate_pass = bool(gh_raw_dominance or gh_gate_structural)
        gh_info["details"]["raw_dominance"] = bool(gh_raw_dominance)
        gh_info["details"]["gate_structural"] = bool(gh_gate_structural)
        gh_info["details"]["gate_pass"] = bool(gh_gate_pass)
        if gh_gate_pass:
            if gh_raw_dominance:
                boosted_score = max(
                    float(gh_info["score"]) + 0.75,
                    float(best_non_gh) + 0.05,
                    gh_base + 0.05,
                )
            else:
                boosted_score = float(gh_info["score"]) + 0.050
            by_earth["Great Hollow"] = {"score": float(boosted_score), "details": gh_info["details"]}

    ranked = [
        (name, float(info["score"]), dict(info["details"]))
        for name, info in by_earth.items()
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def format_layout_debug_lines(result: dict, max_slots: int = 5) -> List[str]:
    details = dict(result.get("details", {}))
    lines = [
        (
            f"layout #{int(result['layout_number'])} earth={result['shifting_earth']} "
            f"total={float(result['total_score']):.6f} normalized={float(details.get('normalized_log_prob', 0.0)):.6f} "
            f"earth={float(result.get('earth_score', 0.0)):.6f} geometry={float(result.get('geometry_prior', 0.0)):+.4f} "
            f"lost={float(result.get('lost_slot_prior', 0.0)):+.4f} hard={float(result.get('hard_constraint_prior', 0.0)):+.4f} "
            f"evergaol={float(result.get('evergaol_prior', 0.0)):+.4f} "
            f"gh_status={float(result.get('great_hollow_status_prior', 0.0)):+.4f}"
        ),
        (
            f"  exact_w={float(details.get('exact_match_weight', 0.0)):.2f} "
            f"mismatch_w={float(details.get('hard_mismatch_weight', 0.0)):.2f} "
            f"unknown_w={float(details.get('unknown_weight', 0.0)):.2f}"
        ),
    ]
    weakest = sorted(
        list(details.get("slot_details", [])),
        key=lambda item: (float(item.get("expected_prob", 0.0)), float(item.get("weighted_log_prob", 0.0))),
    )[:max_slots]
    if weakest:
        lines.append("  weakest expected slots:")
        for item in weakest:
            lines.append(
                "    "
                f"{item['slot_name']} exp={item['expected']} obs={item['observed']} "
                f"p={float(item['expected_prob']):.4f} best={float(item['best_prob']):.4f} "
                f"margin={float(item['margin']):.4f}"
            )
    return lines


def draw_ai_debug(panel_bgr: np.ndarray, slots_out: Sequence[dict]) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    side = min(h, w)
    for slot_obs in slots_out:
        cx = int(round(float(slot_obs["nominal_x_norm"]) * w))
        cy = int(round(float(slot_obs["nominal_y_norm"]) * h))
        px = int(round(float(slot_obs["x_norm"]) * w))
        py = int(round(float(slot_obs["y_norm"]) * h))
        r = max(6, int(round(float(slot_obs["radius_norm"]) * side)))

        cv2.circle(out, (cx, cy), r, (255, 255, 0), 1)

        label = str(slot_obs["label"])
        if label == "unknown":
            color = (0, 0, 255)
        elif label == "empty":
            color = (180, 180, 180)
        else:
            color = (0, 255, 0)

        cv2.circle(out, (px, py), 5, color, -1)

        top_probs = sorted(slot_obs["probs"].items(), key=lambda kv: kv[1], reverse=True)[:2]
        if len(top_probs) >= 2:
            text = f"{slot_obs['slot_name']}: {label} ({top_probs[0][0]} {top_probs[0][1]:.2f} / {top_probs[1][0]} {top_probs[1][1]:.2f})"
        else:
            text = f"{slot_obs['slot_name']}: {label}"

        tx = max(6, min(w - 320, px + 6))
        ty = max(16, min(h - 8, py - 6))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    return out


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    log(f"Using device: {device}")

    backend = load_backend_module(args.backend)
    backend_ctx = make_backend_ctx(backend)
    if args.config_json is not None:
        cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
        apply_config_to_module(backend, cfg)
        log(f"Applied tuned backend config from {args.config_json}")

    model, labels, patch_size, context_scale, min_half_frac = load_model_bundle(args.model, device)
    log(f"Loaded AI slot model from {args.model}")
    log(f"Model labels: {labels}")
    log(f"Patch size={patch_size} context_scale={context_scale} min_half_frac={min_half_frac}")

    package = load_backend_package(backend, args.package)
    log(f"Loaded package: {args.package}")

    crop_method = None
    crop_box = None

    if args.panel_image is not None:
        raw_panel = backend.load_bgr(args.panel_image)
        panel_bgr = normalize_panel_for_recognition_compat(backend, raw_panel)
        full_img = None
        earth_rank = classify_shifting_earth_compat(backend, raw_panel, panel_bgr, package)
        crop_method = "panel_image"
    else:
        if args.monitor is not None:
            full_img = backend.capture_monitor_bgr(args.monitor)
        else:
            full_img = backend.load_bgr(args.image)
        circle, crop_method = detect_map_circle_compat(backend, full_img, backend_ctx)
        if crop_method == "blue_ring":
            crop_box = backend.clip_box(
                backend.estimate_panel_box(circle, backend.DEFAULT_PANEL_SCALE),
                full_img.shape[1],
                full_img.shape[0],
            )
            raw_panel = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
            panel_bgr = normalize_panel_for_recognition_compat(backend, raw_panel)
            earth_rank = classify_shifting_earth_compat(backend, raw_panel, panel_bgr, package)
        else:
            initial_box = refine_panel_box_compat(backend, full_img, circle, backend_ctx)
            crop_box, raw_panel, panel_bgr, metrics, _earth_rank_unused, refine_info = refine_crop_box_by_earth_score_compat(
                backend,
                full_img,
                initial_box,
                package,
                backend_ctx,
            )
            earth_rank = classify_shifting_earth_compat(backend, raw_panel, panel_bgr, package)
            crop_method = "gray_hough_refined"

    if args.debug:
        save_bgr(args.panel_out, panel_bgr)
        log(f"Saved panel used for AI recognition to {args.panel_out}")

    log("Shifting Earth ranking:")
    for name, score, details in earth_rank[:5]:
        low_corr = float(details.get("low_corr", 0.0))
        grad_corr = float(details.get("grad_corr", 0.0))
        source = details.get("source", "legacy")
        log(f"  {name:<14} score={score:.6f} low_corr={low_corr:.4f} grad_corr={grad_corr:.4f} source={source}")

    candidate_earths = [name for name, _score, _details in earth_rank]
    if candidate_earths:
        log(f"Evaluating structural earth cascade across {len(candidate_earths)} families.")

    ranked_layouts = []
    observations_by_earth = {}

    for earth_name in candidate_earths:
        earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
        slots_out, align = ai_classify_slots(
            panel_bgr=panel_bgr,
            backend=backend,
            package=package,
            earth_name=earth_name,
            model=model,
            labels=labels,
            patch_size=patch_size,
            context_scale=context_scale,
            min_half_frac=min_half_frac,
            device=device,
            confidence_threshold=float(args.confidence_threshold),
            margin_threshold=float(args.margin_threshold),
        )
        log(f"AI slot classification for earth {earth_name}: dx={align['dx_norm']:+.4f} dy={align['dy_norm']:+.4f} resp={align['align_response']:.4f}")
        unknown_count = sum(1 for s in slots_out if s["label"] == "unknown")
        log(f"  unknown slots: {unknown_count}/{len(slots_out)}")
        observations_by_earth[earth_name] = {
            "slots": slots_out,
            "align": align,
            "earth_score": float(earth_score),
        }

    geometry_priors = compute_earth_geometry_priors(package, observations_by_earth)
    lost_slot_priors = compute_lost_limveld_slot_priors(package, observations_by_earth)
    hard_constraint_priors = compute_hard_constraint_priors(package, observations_by_earth)
    evergaol_priors = compute_evergaol_priors(panel_bgr, observations_by_earth, device=device)
    great_hollow_status_observations = {}
    if USE_GREAT_HOLLOW_STATUS_PRIOR:
        great_hollow_status_observations = detect_great_hollow_statuses(panel_bgr, package, observations_by_earth)
    for earth_name, info in geometry_priors.items():
        log(
            f"  geometry prior {earth_name}: slot_count={info['slot_count']} "
            f"align={info['align_response']:.4f} prior={info['geometry_prior']:+.4f}"
        )
    for earth_name, info in lost_slot_priors.items():
        log(
            f"  lost-slot prior {earth_name}: lost_slots={info['lost_slot_count']} "
            f"prior={info['lost_slot_prior']:+.4f}"
        )
    for earth_name, info in hard_constraint_priors.items():
        prior = float(info.get("hard_constraint_prior", 0.0))
        if abs(prior) < 1e-9:
            continue
        log(f"  hard constraint prior {earth_name}: prior={prior:+.4f}")
    for earth_name, info in evergaol_priors.items():
        log(f"  evergaol prior {earth_name}: prior={float(info.get('evergaol_prior', 0.0)):+.4f}")
    if USE_GREAT_HOLLOW_STATUS_PRIOR:
        for anchor_type, info in great_hollow_status_observations.items():
            log(
                f"  Great Hollow status {anchor_type}: best={info.get('best_status', '?')} "
                f"score={float(info.get('best_score', 0.0)):.4f} margin={float(info.get('margin', 0.0)):.4f}"
            )

    earth_cascade_rank = rank_shifting_earths_cascade(
        earth_rank,
        geometry_priors,
        lost_slot_priors,
        hard_constraint_priors,
        evergaol_priors,
        great_hollow_status_observations,
    )
    log("Shifting Earth cascade ranking:")
    for name, score, details in earth_cascade_rank[:5]:
        log(
            f"  {name:<14} score={score:.6f} base={float(details.get('base_score', 0.0)):.6f} "
            f"geom={float(details.get('geometry_prior', 0.0)):+.4f} "
            f"lost={float(details.get('lost_slot_prior', 0.0)):+.4f} "
            f"hard={float(details.get('hard_constraint_prior', 0.0)):+.4f} "
            f"ever={float(details.get('evergaol_prior', 0.0)):+.4f}"
        )
    candidate_earths = [name for name, _score, _details in earth_cascade_rank[:1]]
    if candidate_earths:
        log(f"Restricting layout ranking to {candidate_earths[0]} family only after cascade.")

    for earth_name in candidate_earths:
        slots_out = observations_by_earth[earth_name]["slots"]
        earth_score = float(observations_by_earth[earth_name]["earth_score"])
        geometry_prior = float(geometry_priors.get(earth_name, {}).get("geometry_prior", 0.0))
        lost_slot_prior = float(lost_slot_priors.get(earth_name, {}).get("lost_slot_prior", 0.0))
        hard_constraint_prior = float(hard_constraint_priors.get(earth_name, {}).get("hard_constraint_prior", 0.0))
        evergaol_prior = float(evergaol_priors.get(earth_name, {}).get("evergaol_prior", 0.0))
        for layout in package.layouts_by_earth[earth_name]:
            great_hollow_status_prior = 0.0
            great_hollow_status_details = {"great_hollow_status_prior": 0.0, "status_evidence": []}
            if USE_GREAT_HOLLOW_STATUS_PRIOR and earth_name == "Great Hollow":
                great_hollow_status_prior, great_hollow_status_details = compute_great_hollow_status_prior(
                    layout,
                    great_hollow_status_observations,
                )
            total_score, details = score_layout_ai(
                slots_out,
                layout["slot_labels"],
                earth_score,
                earth_name=earth_name,
                geometry_prior=geometry_prior,
                lost_slot_prior=lost_slot_prior,
                hard_constraint_prior=hard_constraint_prior,
                evergaol_prior=evergaol_prior,
                great_hollow_status_prior=great_hollow_status_prior,
            )
            details["status_evidence"] = list(great_hollow_status_details.get("status_evidence", []))
            ranked_layouts.append({
                "layout_number": int(layout["layout_number"]),
                "shifting_earth": earth_name,
                "total_score": float(total_score),
                "earth_score": float(earth_score),
                "geometry_prior": float(geometry_prior),
                "lost_slot_prior": float(lost_slot_prior),
                "hard_constraint_prior": float(hard_constraint_prior),
                "evergaol_prior": float(evergaol_prior),
                "great_hollow_status_prior": float(great_hollow_status_prior),
                "details": details,
            })

    ranked_layouts.sort(key=lambda item: item["total_score"], reverse=True)
    best_earth_name = ranked_layouts[0]["shifting_earth"] if ranked_layouts else (candidate_earths[0] if candidate_earths else None)
    best_slots_debug = observations_by_earth.get(best_earth_name, {}).get("slots")

    if args.debug and best_slots_debug is not None:
        dbg = draw_ai_debug(panel_bgr, best_slots_debug)
        save_bgr(args.ai_debug_out, dbg)
        log(f"Saved AI slot debug image to {args.ai_debug_out}")
    if args.debug and evergaol_priors:
        evergaol_dbg = draw_evergaol_debug(panel_bgr, evergaol_priors)
        save_bgr(args.evergaol_debug_out, evergaol_dbg)
        log(f"Saved Evergaol debug image to {args.evergaol_debug_out}")

    print()
    print(f"Top {min(args.top, len(ranked_layouts))} layouts (AI slot classifier):")
    for idx, item in enumerate(ranked_layouts[:args.top], start=1):
        print(
            f"{idx:>2}. layout #{item['layout_number']:<3} earth={item['shifting_earth']:<14} "
            f"score={item['total_score']:.6f} earth_score={item['earth_score']:.6f} "
            f"gh_status={float(item.get('great_hollow_status_prior', 0.0)):+.4f}"
        )

    by_layout_number = {int(item["layout_number"]): item for item in ranked_layouts}
    for left_id, right_id in SPECIAL_LAYOUT_COMPARISONS:
        left = by_layout_number.get(int(left_id))
        right = by_layout_number.get(int(right_id))
        if left is None or right is None:
            continue
        print()
        print(f"Comparison layout #{left_id} vs #{right_id}:")
        for line in format_layout_debug_lines(left):
            print(line)
        for line in format_layout_debug_lines(right):
            print(line)
        print(f"  score delta ({left_id}-{right_id}) = {float(left['total_score']) - float(right['total_score']):+.6f}")

    payload = {
        "crop_method": crop_method,
        "crop_box": None if crop_box is None else crop_box.__dict__,
        "earth_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_rank],
        "earth_cascade_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_cascade_rank],
        "candidate_earths": candidate_earths,
        "best_earth_for_debug": best_earth_name,
        "ai_slots_debug": best_slots_debug,
        "geometry_priors": geometry_priors,
        "lost_slot_priors": lost_slot_priors,
        "hard_constraint_priors": hard_constraint_priors,
        "evergaol_priors": evergaol_priors,
        "great_hollow_status_observations": great_hollow_status_observations,
        "results": ranked_layouts,
        "model_path": str(args.model),
        "evergaol_debug_out": str(args.evergaol_debug_out),
        "labels": list(labels),
        "confidence_threshold": float(args.confidence_threshold),
        "margin_threshold": float(args.margin_threshold),
    }
    ensure_parent(args.json_out)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Saved AI inference JSON report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
