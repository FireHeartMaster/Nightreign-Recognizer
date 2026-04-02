#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CMP_SIZE = 512
DEFAULT_HOUGH_REGION = "right_half"
DEFAULT_HOUGH_SCALE = 0.5
DEFAULT_PANEL_SCALE = 1.12
DEFAULT_SEARCH_RADIUS = 80
DEFAULT_PAD = 0
DEFAULT_TEMPLATE_SCALES = (0.86, 0.94, 1.00, 1.08, 1.16)

LANDMARK_TYPES = ["Church", "Fort", "Great Church", "Main Encampment", "Ruins", "Sorcerer's Rise"]
TYPE_CONFIG = {
    "Church": {
        "window_frac": 0.110,
        "empty_threshold": 0.28,
        "accept": 0.40,
        "margin": 0.025,
        "rgb_weight": 0.16,
        "gray_weight": 0.22,
        "edge_weight": 0.62,
    },
    "Fort": {
        "window_frac": 0.090,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.22,
        "edge_weight": 0.40,
    },
    "Great Church": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.20,
        "edge_weight": 0.42,
    },
    "Main Encampment": {
        "window_frac": 0.100,
        "empty_threshold": 0.29,
        "accept": 0.40,
        "margin": 0.028,
        "rgb_weight": 0.30,
        "gray_weight": 0.22,
        "edge_weight": 0.48,
    },
    "Ruins": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.42,
        "margin": 0.028,
        "rgb_weight": 0.34,
        "gray_weight": 0.18,
        "edge_weight": 0.48,
    },
    "Sorcerer's Rise": {
        "window_frac": 0.118,
        "empty_threshold": 0.26,
        "accept": 0.36,
        "margin": 0.020,
        "rgb_weight": 0.14,
        "gray_weight": 0.22,
        "edge_weight": 0.64,
    },
}
EARTH_SECONDARY_THRESHOLD = 0.025


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


@dataclass
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class SlotObservation:
    slot_id: int
    slot_name: str
    x_norm: float
    y_norm: float
    label: str
    best_score: float
    scores: Dict[str, float]
    best_x_norm: float
    best_y_norm: float
    expected_types: List[str]


@dataclass
class MatchResult:
    layout_number: int
    shifting_earth: str
    total_score: float
    earth_score: float
    details: Dict[str, float]


class RecognitionPackage:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata_json"]))
            self.earth_names = [str(x) for x in data["earth_names"]]
            self.terrain_low = data["terrain_low"]
            self.terrain_grad = data["terrain_grad"]
            self.landmark_types = [str(x) for x in data["landmark_types"]]
            self.icon_names = [str(x) for x in data["icon_names"]]
            self.icon_templates = data["icon_templates"]
            self.icon_templates_gray = data["icon_templates_gray"]
            self.icon_templates_edge = data["icon_templates_edge"]
            self.icon_actual_sizes = data["icon_actual_sizes"]
        self.earth_to_idx = {name: idx for idx, name in enumerate(self.earth_names)}
        self.type_to_idx = {name: idx for idx, name in enumerate(self.landmark_types)}
        self.layouts = list(self.metadata["layouts"])
        self.layouts_by_earth: Dict[str, List[dict]] = {name: [] for name in self.earth_names}
        for item in self.layouts:
            self.layouts_by_earth[str(item["shifting_earth"])].append(item)
        self.slots_by_earth: Dict[str, List[dict]] = {
            key: list(value) for key, value in self.metadata["slots_by_earth"].items()
        }
        self._template_variants_cache: Dict[str, List[dict]] = {}

    def template_variants(self, type_name: str) -> List[dict]:
        cached = self._template_variants_cache.get(type_name)
        if cached is not None:
            return cached
        type_idx = self.type_to_idx[type_name]
        rgba = self.icon_templates[type_idx]
        gray_t = self.icon_templates_gray[type_idx]
        edge_t = self.icon_templates_edge[type_idx]
        x0, y0, x1, y1 = alpha_bbox(rgba)
        rgba = rgba[y0:y1, x0:x1]
        gray_t = gray_t[y0:y1, x0:x1]
        edge_t = edge_t[y0:y1, x0:x1]
        rgb_base = rgba[:, :, :3]
        mask_base = rgba[:, :, 3]
        variants: List[dict] = []
        for scale in DEFAULT_TEMPLATE_SCALES:
            new_w = max(8, int(round(rgb_base.shape[1] * scale)))
            new_h = max(8, int(round(rgb_base.shape[0] * scale)))
            rgb_t = cv2.resize(rgb_base, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray_rs = cv2.resize(gray_t, (new_w, new_h), interpolation=cv2.INTER_AREA)
            edge_rs = cv2.resize(edge_t, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_rs = cv2.resize(mask_base, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_bin = np.where(mask_rs > 8, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(mask_bin)) == 0:
                continue
            variants.append({
                "scale": scale,
                "rgb": rgb_t,
                "gray": gray_rs,
                "edge": edge_rs,
                "mask": mask_bin,
                "w": new_w,
                "h": new_h,
                "cx": new_w // 2,
                "cy": new_h // 2,
            })
        self._template_variants_cache[type_name] = variants
        return variants


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototype slot-based recognizer: terrain first, then per-slot landmark labeling with real empty slots.")
    p.add_argument("--package", type=Path, required=True, help="Recognition package generated by the slot-based renderer.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Full screenshot image file to process.")
    source.add_argument("--panel-image", type=Path, help="Already-cropped square map panel image to process.")
    source.add_argument("--monitor", type=int, help="Capture this monitor index with mss (1-based).")
    p.add_argument("--top", type=int, default=20, help="How many ranked layouts to print.")
    p.add_argument("--debug", action="store_true", help="Save debug images showing the exact panel used for recognition and the detected landmark labels.")
    p.add_argument("--crop-out", type=Path, default=Path("recognition_panel_used.png"), help="Path to save the exact cropped map panel used for recognition when --debug is enabled.")
    p.add_argument("--detections-out", type=Path, default=Path("recognition_panel_used_with_slot_labels.png"), help="Path to save the recognition panel with detected landmark labels drawn on it when --debug is enabled.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to save a JSON report.")
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def ensure_parent(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_bgr(path: Optional[Path], img: np.ndarray) -> None:
    if path is not None:
        ensure_parent(path)
        cv2.imwrite(str(path), img)


def capture_monitor_bgr(monitor_index: int) -> np.ndarray:
    try:
        import mss  # type: ignore
    except Exception as exc:
        raise RuntimeError("Screen capture requires the mss package to be installed.") from exc
    with mss.mss() as sct:
        if monitor_index < 1 or monitor_index >= len(sct.monitors):
            available = max(0, len(sct.monitors) - 1)
            raise ValueError(f"Monitor index {monitor_index} is not available. Detected {available} screen(s).")
        monitor = sct.monitors[monitor_index]
        shot = np.array(sct.grab(monitor), dtype=np.uint8)
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def clip_box(box: CropBox, w: int, h: int) -> CropBox:
    return CropBox(
        max(0, min(w - 1, box.x0)),
        max(1, min(h - 1, box.y0)),
        max(1, min(w, box.x1)),
        max(1, min(h, box.y1)),
    )


def estimate_panel_box(circle: Circle, panel_scale: float) -> CropBox:
    side = 2.0 * circle.r * panel_scale
    half = side / 2.0
    return CropBox(
        int(round(circle.cx - half)),
        int(round(circle.cy - half)),
        int(round(circle.cx + half)),
        int(round(circle.cy + half)),
    )


def detect_map_circle_blue(full_img: np.ndarray) -> Optional[Circle]:
    h, w = full_img.shape[:2]
    fallback = Circle(w * 0.76, h * 0.50, min(h, w) * 0.32)
    if DEFAULT_HOUGH_REGION == "right_half":
        x0, y0, x1, y1 = w // 2, 0, w, h
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    roi = full_img[y0:y1, x0:x1]
    scale = DEFAULT_HOUGH_SCALE
    small_w = max(1, int(round((x1 - x0) * scale)))
    small_h = max(1, int(round((y1 - y0) * scale)))
    roi_small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
    lower = np.array([88, 20, 70], dtype=np.uint8)
    upper = np.array([132, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (9, 9), 2.0)
    min_radius = int(min(small_h, small_w) * 0.18)
    max_radius = int(min(small_h, small_w) * 0.44)
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, min(small_h, small_w) // 3),
        param1=100,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or len(circles[0]) == 0:
        return None
    best = None
    best_score = None
    for c in circles[0]:
        cx, cy, r = map(float, c)
        full_cx = x0 + (cx / scale)
        full_cy = y0 + (cy / scale)
        full_r = r / scale
        score = abs(full_cx - fallback.cx) / w + abs(full_cy - fallback.cy) / h - 0.40 * (full_r / min(h, w))
        if best_score is None or score < best_score:
            best_score = score
            best = Circle(full_cx, full_cy, full_r)
    if best is None:
        return None
    center_dx = best.cx - fallback.cx
    center_dy = best.cy - fallback.cy
    radius_ratio = best.r / fallback.r if fallback.r > 1e-6 else 1.0
    if abs(center_dx) <= w * 0.03 and abs(center_dy) <= h * 0.08 and 0.92 <= radius_ratio <= 1.10:
        return best
    return None


def detect_map_circle_gray(full_img: np.ndarray) -> Circle:
    h, w = full_img.shape[:2]
    fallback = Circle(w * 0.76, h * 0.50, min(h, w) * 0.32)
    if DEFAULT_HOUGH_REGION == "right_half":
        x0, y0, x1, y1 = w // 2, 0, w, h
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    roi = full_img[y0:y1, x0:x1]
    scale = DEFAULT_HOUGH_SCALE
    small_w = max(1, int(round((x1 - x0) * scale)))
    small_h = max(1, int(round((y1 - y0) * scale)))
    roi_small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    min_radius = int(min(small_h, small_w) * 0.18)
    max_radius = int(min(small_h, small_w) * 0.42)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, min(small_h, small_w) // 3),
        param1=120,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or len(circles[0]) == 0:
        return fallback
    best = None
    best_score = None
    for c in circles[0]:
        cx, cy, r = map(float, c)
        full_cx = x0 + (cx / scale)
        full_cy = y0 + (cy / scale)
        full_r = r / scale
        score = abs(full_cx - fallback.cx) / w + abs(full_cy - fallback.cy) / h - 0.30 * (full_r / min(h, w))
        if best_score is None or score < best_score:
            best_score = score
            best = Circle(full_cx, full_cy, full_r)
    chosen = best if best is not None else fallback
    center_dx = chosen.cx - fallback.cx
    center_dy = chosen.cy - fallback.cy
    radius_ratio = chosen.r / fallback.r if fallback.r > 1e-6 else 1.0
    if abs(center_dx) > w * 0.020 or abs(center_dy) > h * 0.050 or not (0.95 <= radius_ratio <= 1.08):
        return fallback
    return Circle(
        fallback.cx + center_dx * 0.25,
        fallback.cy + center_dy * 0.25,
        fallback.r + (chosen.r - fallback.r) * 0.25,
    )


def refine_vertical_edge(gray: np.ndarray, gradx: np.ndarray, x_est: int, y0: int, y1: int, search_radius: int) -> int:
    h, w = gray.shape
    x_start = max(0, x_est - search_radius)
    x_end = min(w - 1, x_est + search_radius)
    best_x = x_est
    best_score = None
    for x in range(x_start, x_end + 1):
        col = gray[y0:y1, max(0, x - 1):min(w, x + 2)]
        gx = np.abs(gradx[y0:y1, x])
        score = float((col > 150).sum()) + 3.0 * float(gx.mean())
        if best_score is None or score > best_score:
            best_score = score
            best_x = x
    return best_x


def refine_horizontal_edge(gray: np.ndarray, grady: np.ndarray, y_est: int, x0: int, x1: int, search_radius: int) -> int:
    h, w = gray.shape
    y_start = max(0, y_est - search_radius)
    y_end = min(h - 1, y_est + search_radius)
    best_y = y_est
    best_score = None
    for y in range(y_start, y_end + 1):
        row = gray[max(0, y - 1):min(h, y + 2), x0:x1]
        gy = np.abs(grady[y, x0:x1])
        score = float((row > 150).sum()) + 3.0 * float(gy.mean())
        if best_score is None or score > best_score:
            best_score = score
            best_y = y
    return best_y


def refine_panel_box(img_bgr: np.ndarray, circle: Circle) -> CropBox:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    est = clip_box(estimate_panel_box(circle, DEFAULT_PANEL_SCALE), w, h)
    band_y0 = max(0, int(round(circle.cy - circle.r * 1.18)))
    band_y1 = min(h, int(round(circle.cy + circle.r * 1.18)))
    band_x0 = max(0, int(round(circle.cx - circle.r * 1.18)))
    band_x1 = min(w, int(round(circle.cx + circle.r * 1.18)))
    left = refine_vertical_edge(gray, gradx, est.x0, band_y0, band_y1, DEFAULT_SEARCH_RADIUS)
    right = refine_vertical_edge(gray, gradx, est.x1, band_y0, band_y1, DEFAULT_SEARCH_RADIUS)
    top = refine_horizontal_edge(gray, grady, est.y0, band_x0, band_x1, DEFAULT_SEARCH_RADIUS)
    bottom = refine_horizontal_edge(gray, grady, est.y1, band_x0, band_x1, DEFAULT_SEARCH_RADIUS)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    side = max(32, int(round(((right - left) + (bottom - top)) / 2.0)))
    half = side / 2.0
    box = clip_box(
        CropBox(
            int(round(center_x - half)) - DEFAULT_PAD,
            int(round(center_y - half)) - DEFAULT_PAD,
            int(round(center_x + half)) + DEFAULT_PAD,
            int(round(center_y + half)) + DEFAULT_PAD,
        ),
        w, h,
    )
    side2 = min(box.x1 - box.x0, box.y1 - box.y0)
    return clip_box(CropBox(box.x0, box.y0, box.x0 + side2, box.y0 + side2), w, h)


def detect_map_circle(full_img: np.ndarray) -> Tuple[Circle, str]:
    blue = detect_map_circle_blue(full_img)
    if blue is not None:
        return blue, "blue_ring"
    return detect_map_circle_gray(full_img), "gray_hough"


def preprocess_for_similarity(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (CMP_SIZE, CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 6.0)
    return gray


def preprocess_for_gradients(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (CMP_SIZE, CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
    return mag


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd <= 1e-6:
        sd = 1.0
    return (arr - mu) / sd


def corr(a: np.ndarray, b: np.ndarray) -> float:
    va = a.astype(np.float32).ravel()
    vb = b.astype(np.float32).ravel()
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


def classify_shifting_earth(panel_bgr: np.ndarray, package: RecognitionPackage) -> List[Tuple[str, float, Dict[str, float]]]:
    shot_low = preprocess_for_similarity(panel_bgr)
    shot_grad = preprocess_for_gradients(panel_bgr)
    shot_low_n = normalize(shot_low)
    shot_grad_n = normalize(shot_grad)
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for idx, earth_name in enumerate(package.earth_names):
        low_corr = corr(shot_low_n, normalize(package.terrain_low[idx]))
        grad_corr = corr(shot_grad_n, normalize(package.terrain_grad[idx]))
        score = 0.65 * low_corr + 0.35 * grad_corr
        ranked.append((earth_name, score, {"low_corr": low_corr, "grad_corr": grad_corr}))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def alpha_bbox(rgba: np.ndarray) -> Tuple[int, int, int, int]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_panel_edge(panel_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    if float(mag.max()) > 1e-6:
        mag = mag / float(mag.max()) * 255.0
    return mag.astype(np.uint8)


def build_panel_representations(panel_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "rgb": cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB),
        "gray": cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY),
        "edge": build_panel_edge(panel_bgr),
    }


def estimate_global_translation(panel_bgr: np.ndarray, package: RecognitionPackage, earth_name: str) -> Tuple[float, float, float]:
    idx = package.earth_to_idx[earth_name]
    shot_low = preprocess_for_similarity(panel_bgr).astype(np.float32)
    ref_low = package.terrain_low[idx].astype(np.float32)
    shift, response = cv2.phaseCorrelate(ref_low, shot_low)
    dx_norm = float(np.clip(shift[0] / CMP_SIZE, -0.04, 0.04))
    dy_norm = float(np.clip(shift[1] / CMP_SIZE, -0.05, 0.05))
    return dx_norm, dy_norm, float(response)


def local_template_score(panel_repr: Dict[str, np.ndarray], package: RecognitionPackage, type_name: str, cx: int, cy: int, search_radius: int) -> Tuple[float, Tuple[int, int]]:
    cfg = TYPE_CONFIG[type_name]
    panel_rgb = panel_repr["rgb"]
    panel_gray = panel_repr["gray"]
    panel_edge = panel_repr["edge"]
    h, w = panel_gray.shape[:2]
    best_score = -1.0
    best_xy = (cx, cy)
    for variant in package.template_variants(type_name):
        tw = int(variant["w"])
        th = int(variant["h"])
        halfw = search_radius + tw // 2 + 2
        halfh = search_radius + th // 2 + 2
        x0 = max(0, cx - halfw)
        y0 = max(0, cy - halfh)
        x1 = min(w, cx + halfw)
        y1 = min(h, cy + halfh)
        if x1 - x0 <= tw or y1 - y0 <= th:
            continue
        patch_rgb = panel_rgb[y0:y1, x0:x1]
        patch_gray = panel_gray[y0:y1, x0:x1]
        patch_edge = panel_edge[y0:y1, x0:x1]
        rgb_score = cv2.matchTemplate(patch_rgb, variant["rgb"], cv2.TM_CCORR_NORMED, mask=variant["mask"])
        gray_score = cv2.matchTemplate(patch_gray, variant["gray"], cv2.TM_CCORR_NORMED, mask=variant["mask"])
        edge_score = cv2.matchTemplate(patch_edge, variant["edge"], cv2.TM_CCORR_NORMED, mask=variant["mask"])
        combined = (
            float(cfg["rgb_weight"]) * rgb_score
            + float(cfg["gray_weight"]) * gray_score
            + float(cfg["edge_weight"]) * edge_score
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(combined)
        if float(max_val) > best_score:
            best_score = float(max_val)
            best_xy = (x0 + int(max_loc[0]) + int(variant["cx"]), y0 + int(max_loc[1]) + int(variant["cy"]))
    return best_score, best_xy


def classify_slots_for_earth(panel_bgr: np.ndarray, package: RecognitionPackage, earth_name: str) -> Tuple[List[SlotObservation], Dict[str, float]]:
    dx_norm, dy_norm, align_response = estimate_global_translation(panel_bgr, package, earth_name)
    panel_repr = build_panel_representations(panel_bgr)
    h, w = panel_bgr.shape[:2]
    side = min(h, w)
    observations: List[SlotObservation] = []
    for slot in package.slots_by_earth[earth_name]:
        base_cx = int(round((float(slot["x_norm"]) + dx_norm) * w))
        base_cy = int(round((float(slot["y_norm"]) + dy_norm) * h))
        base_cx = max(0, min(w - 1, base_cx))
        base_cy = max(0, min(h - 1, base_cy))
        search_radius = max(
            int(round(float(slot["radius_norm"]) * side * 1.25)),
            max(int(round(TYPE_CONFIG[t]["window_frac"] * side * 0.55)) for t in slot.get("allowed_types", LANDMARK_TYPES))
        )
        type_scores: Dict[str, float] = {}
        type_positions: Dict[str, Tuple[int, int]] = {}
        candidate_types = list(slot.get("allowed_types", LANDMARK_TYPES)) or LANDMARK_TYPES
        for type_name in candidate_types:
            score, best_xy = local_template_score(panel_repr, package, type_name, base_cx, base_cy, search_radius)
            type_scores[type_name] = score
            type_positions[type_name] = best_xy
        if type_scores:
            ordered = sorted(type_scores.items(), key=lambda kv: kv[1], reverse=True)
            best_label, best_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        else:
            best_label, best_score, second_score = ("empty", -1.0, -1.0)
        cfg = TYPE_CONFIG.get(best_label, TYPE_CONFIG["Ruins"])
        if best_score < float(cfg["empty_threshold"]):
            final_label = "empty"
            best_xy = (base_cx, base_cy)
        elif best_score < float(cfg["accept"]) or (best_score - second_score) < float(cfg["margin"]):
            final_label = "unknown"
            best_xy = type_positions.get(best_label, (base_cx, base_cy))
        else:
            final_label = best_label
            best_xy = type_positions.get(best_label, (base_cx, base_cy))
        observations.append(
            SlotObservation(
                slot_id=int(slot["slot_id"]),
                slot_name=str(slot["slot_name"]),
                x_norm=float(best_xy[0] / w),
                y_norm=float(best_xy[1] / h),
                label=final_label,
                best_score=float(best_score),
                scores={k: float(v) for k, v in type_scores.items()},
                best_x_norm=float(best_xy[0] / w),
                best_y_norm=float(best_xy[1] / h),
                expected_types=list(candidate_types),
            )
        )
    return observations, {"dx_norm": dx_norm, "dy_norm": dy_norm, "align_response": align_response}


def score_layout(slot_observations: Sequence[SlotObservation], layout_slot_labels: Sequence[str], slot_defs: Sequence[dict], earth_score: float, earth_name: str, layout_number: int) -> MatchResult:
    total = 0.0
    exact_matches = 0.0
    hard_mismatches = 0.0
    unknown_count = 0.0
    for obs, exp, slot in zip(slot_observations, layout_slot_labels, slot_defs):
        weight = float(slot.get("weight", 1.0))
        type_scores = obs.scores
        max_nonempty = max(type_scores.values()) if type_scores else 0.0
        if exp == "empty":
            contrib = 0.85 - 2.20 * max_nonempty
            if obs.label == "empty":
                contrib += 0.18
        else:
            exp_score = float(type_scores.get(exp, 0.0))
            best_other = max((score for t, score in type_scores.items() if t != exp), default=0.0)
            contrib = 2.4 * exp_score - 1.9 * best_other
            if obs.label == exp:
                contrib += 0.55
                exact_matches += weight
            elif obs.label == "unknown":
                contrib += 0.0
                unknown_count += weight
            elif obs.label == "empty":
                contrib -= 0.95
                hard_mismatches += weight
            elif obs.label != exp:
                contrib -= 0.85
                hard_mismatches += weight
        total += weight * contrib
    total += 0.35 * earth_score
    return MatchResult(
        layout_number=int(layout_number),
        shifting_earth=earth_name,
        total_score=float(total),
        earth_score=float(earth_score),
        details={
            "exact_match_weight": float(exact_matches),
            "hard_mismatch_weight": float(hard_mismatches),
            "unknown_weight": float(unknown_count),
        },
    )


def draw_slot_debug(panel_bgr: np.ndarray, slot_observations: Sequence[SlotObservation], slot_defs: Sequence[dict]) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    for obs, slot in zip(slot_observations, slot_defs):
        cx = int(round(float(slot["x_norm"]) * w))
        cy = int(round(float(slot["y_norm"]) * h))
        r = max(6, int(round(float(slot["radius_norm"]) * min(h, w))))
        cv2.circle(out, (cx, cy), r, (255, 255, 0), 1)
        px = int(round(obs.best_x_norm * w))
        py = int(round(obs.best_y_norm * h))
        color = (0, 255, 0) if obs.label not in {"empty", "unknown"} else ((0, 180, 255) if obs.label == "unknown" else (180, 180, 180))
        cv2.circle(out, (px, py), 4, color, -1)
        label_text = f"{slot['slot_name']}: {obs.label}"
        text_x = max(6, min(w - 220, px + 6))
        text_y = max(16, min(h - 8, py - 6))
        cv2.putText(out, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def run_recognition(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    package = RecognitionPackage(args.package)
    log(f"Loaded package: {args.package}")

    if args.panel_image is not None:
        panel_bgr = load_bgr(args.panel_image)
        crop_box = None
        crop_method = "panel_image"
    else:
        full_img = capture_monitor_bgr(args.monitor) if args.monitor is not None else load_bgr(args.image)
        crop_circle, crop_method = detect_map_circle(full_img)
        if crop_method == "blue_ring":
            crop_box = clip_box(estimate_panel_box(crop_circle, DEFAULT_PANEL_SCALE), full_img.shape[1], full_img.shape[0])
        else:
            crop_box = refine_panel_box(full_img, crop_circle)
        panel_bgr = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
    if args.debug:
        save_bgr(args.crop_out, panel_bgr)
        log(f"Saved panel used for recognition to {args.crop_out}")

    earth_rank = classify_shifting_earth(panel_bgr, package)
    log("Shifting Earth ranking:")
    for name, score, details in earth_rank[:5]:
        log(f"  {name:<14} score={score:.6f} low_corr={details['low_corr']:.4f} grad_corr={details['grad_corr']:.4f}")

    candidate_earths = [earth_rank[0][0]]
    if len(earth_rank) > 1 and (earth_rank[0][1] - earth_rank[1][1]) < EARTH_SECONDARY_THRESHOLD:
        candidate_earths.append(earth_rank[1][0])

    all_results: List[MatchResult] = []
    all_observation_sets: Dict[str, Tuple[List[SlotObservation], Dict[str, float]]] = {}
    for earth_name in candidate_earths:
        earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
        log(f"Classifying slots for earth {earth_name}...")
        observations, align_info = classify_slots_for_earth(panel_bgr, package, earth_name)
        all_observation_sets[earth_name] = (observations, align_info)
        log(f"  alignment dx={align_info['dx_norm']:+.4f} dy={align_info['dy_norm']:+.4f} resp={align_info['align_response']:.4f}")
        for layout in package.layouts_by_earth[earth_name]:
            result = score_layout(observations, layout["slot_labels"], package.slots_by_earth[earth_name], earth_score, earth_name, int(layout["layout_number"]))
            all_results.append(result)

    all_results.sort(key=lambda item: item.total_score, reverse=True)
    best_earth = all_results[0].shifting_earth if all_results else candidate_earths[0]
    best_observations, best_align = all_observation_sets[best_earth]
    if args.debug:
        debug_img = draw_slot_debug(panel_bgr, best_observations, package.slots_by_earth[best_earth])
        save_bgr(args.detections_out, debug_img)
        log(f"Saved slot debug image to {args.detections_out}")

    print()
    print(f"Top {min(args.top, len(all_results))} layouts:")
    for rank, item in enumerate(all_results[:args.top], start=1):
        print(
            f"{rank:>2}. layout #{item.layout_number:<3} earth={item.shifting_earth:<14} "
            f"score={item.total_score:.6f} earth_score={item.earth_score:.6f} "
            f"exactW={item.details['exact_match_weight']:.2f} mismW={item.details['hard_mismatch_weight']:.2f} unkW={item.details['unknown_weight']:.2f}"
        )

    payload = {
        "crop_method": crop_method,
        "crop_box": None if crop_box is None else crop_box.__dict__,
        "earth_ranking": [{"name": name, "score": score, "details": details} for name, score, details in earth_rank],
        "candidate_earths": candidate_earths,
        "observations_by_earth": {
            earth: {
                "align": align,
                "slots": [
                    {
                        "slot_id": obs.slot_id,
                        "slot_name": obs.slot_name,
                        "label": obs.label,
                        "best_score": obs.best_score,
                        "scores": obs.scores,
                        "x_norm": obs.best_x_norm,
                        "y_norm": obs.best_y_norm,
                        "expected_types": obs.expected_types,
                    }
                    for obs in obs_list
                ],
            }
            for earth, (obs_list, align) in all_observation_sets.items()
        },
        "results": [
            {
                "layout_number": item.layout_number,
                "shifting_earth": item.shifting_earth,
                "total_score": item.total_score,
                "earth_score": item.earth_score,
                "details": item.details,
            }
            for item in all_results
        ],
        "timing_seconds": time.perf_counter() - t0,
    }
    if args.json_out is not None:
        ensure_parent(args.json_out)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"Saved JSON report to {args.json_out}")
    log(f"Total runtime: {fmt_seconds(payload['timing_seconds'])}")
    return payload


def main() -> int:
    args = parse_args()
    run_recognition(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
