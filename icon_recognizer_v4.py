import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

BASE_REF_SIZE = 2048.0
TOP_LEFT_XY_NORM = (635.0 / BASE_REF_SIZE, 538.0 / BASE_REF_SIZE)
BOTTOM_RIGHT_XY_NORM = (1363.0 / BASE_REF_SIZE, 1400.0 / BASE_REF_SIZE)
LEGACY_PANEL_SIZE = 1536.0
LEGACY_BORDER_CROP_PX = 256.0
CROPPED_PANEL_SIZE = LEGACY_PANEL_SIZE - (2.0 * LEGACY_BORDER_CROP_PX)
AUTO_CROPPED_SIZE_TOL = 64.0

CLASS_MAP = {
    "top_left": {"allowed": ["Fire", "Magic"], "xy_norm": TOP_LEFT_XY_NORM},
    "bottom_right": {"allowed": ["Rot", "Holy"], "xy_norm": BOTTOM_RIGHT_XY_NORM},
    "ruins": {"allowed": ["Poison", "Bleed"], "xy_norm": None},
    "great_church_top_right": {"allowed": ["Lightning", "Sleep"], "xy_norm": None},
}

REF1_LABELS = {"top_left": "Fire", "bottom_right": "Rot"}
REF2_LABELS = {"top_left": "Magic", "bottom_right": "Holy"}
RUINS_CANDIDATE_XY_NORMS = [
    (0.207813, 0.592969),
    (0.325000, 0.526563),
    (0.334375, 0.823438),
    (0.640625, 0.524219),
]
POISON_ICON_CANDIDATES = [
    Path("POI_icons") / "Status_Poison.png",
    Path(__file__).resolve().parent / "POI_icons" / "Status_Poison.png",
]
BLEED_ICON_CANDIDATES = [
    Path("POI_icons") / "Status_Bleed.png",
    Path(__file__).resolve().parent / "POI_icons" / "Status_Bleed.png",
]
GREAT_CHURCH_TOP_RIGHT_CENTER_XY_NORM = (0.89, 0.38)
GREAT_CHURCH_CANDIDATE_XY_NORMS = [
    # User-calibrated anchor (percent space: x=89%, y=62% from bottom => y=38% from top).
    GREAT_CHURCH_TOP_RIGHT_CENTER_XY_NORM,
    # Tight neighborhood for minor crop/align variance.
    (0.86, 0.36),
    (0.88, 0.37),
    (0.90, 0.39),
    (0.92, 0.40),
    (0.94, 0.41),
]
LIGHTNING_ICON_CANDIDATES = [
    Path("POI_icons") / "Status_Lightning.png",
    Path(__file__).resolve().parent / "POI_icons" / "Status_Lightning.png",
]
SLEEP_ICON_CANDIDATES = [
    Path("POI_icons") / "Status_Sleep.png",
    Path(__file__).resolve().parent / "POI_icons" / "Status_Sleep.png",
]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def get_legacy_crop_config():
    left = _env_float("ICON_RECOGNIZER_LEGACY_CROP_LEFT", LEGACY_BORDER_CROP_PX)
    right = _env_float("ICON_RECOGNIZER_LEGACY_CROP_RIGHT", LEGACY_BORDER_CROP_PX)
    top = _env_float("ICON_RECOGNIZER_LEGACY_CROP_TOP", LEGACY_BORDER_CROP_PX)
    bottom = _env_float("ICON_RECOGNIZER_LEGACY_CROP_BOTTOM", LEGACY_BORDER_CROP_PX)
    crop_w = max(1.0, LEGACY_PANEL_SIZE - left - right)
    crop_h = max(1.0, LEGACY_PANEL_SIZE - top - bottom)
    return left, right, top, bottom, crop_w, crop_h


def remap_norm_legacy_to_cropped(xy_norm):
    left, _right, top, _bottom, crop_w, crop_h = get_legacy_crop_config()
    x_legacy = float(xy_norm[0]) * LEGACY_PANEL_SIZE
    y_legacy = float(xy_norm[1]) * LEGACY_PANEL_SIZE
    x_crop = (x_legacy - left) / crop_w
    y_crop = (y_legacy - top) / crop_h
    return float(np.clip(x_crop, 0.0, 1.0)), float(np.clip(y_crop, 0.0, 1.0))


def resolve_xy_norm_for_image(img: np.ndarray, xy_norm):
    h, w = img.shape[:2]
    _left, _right, _top, _bottom, crop_w, crop_h = get_legacy_crop_config()
    if abs(float(w) - crop_w) <= AUTO_CROPPED_SIZE_TOL and abs(float(h) - crop_h) <= AUTO_CROPPED_SIZE_TOL:
        return remap_norm_legacy_to_cropped(xy_norm)
    if abs(float(min(h, w)) - CROPPED_PANEL_SIZE) <= AUTO_CROPPED_SIZE_TOL:
        # Backward-compat with old square 1024 path.
        return remap_norm_legacy_to_cropped(xy_norm)
    return xy_norm


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_to_match(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    if reference.shape[:2] == query.shape[:2]:
        return query
    return cv2.resize(query, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def align_image(reference: np.ndarray, query: np.ndarray):
    original_query = query.copy()
    resized_query = resize_to_match(reference, query)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    qry_gray = cv2.cvtColor(resized_query, cv2.COLOR_BGR2GRAY)

    try:
        detector = cv2.SIFT_create(nfeatures=8000)
        norm_type = cv2.NORM_L2
        ratio = 0.75
    except Exception:
        detector = cv2.ORB_create(nfeatures=10000)
        norm_type = cv2.NORM_HAMMING
        ratio = 0.85

    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    kp_qry, des_qry = detector.detectAndCompute(qry_gray, None)

    if des_ref is None or des_qry is None:
        raise RuntimeError("Could not extract enough features for alignment.")

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(des_qry, des_ref, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError(f"Not enough good matches for homography: {len(good)}")

    src_pts = np.float32([kp_qry[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    aligned = cv2.warpPerspective(
        resized_query,
        H,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, H, original_query, resized_query


def get_scaled_params(img: np.ndarray, xy_norm):
    h, w = img.shape[:2]
    xy = resolve_xy_norm_for_image(img, xy_norm)
    x = int(round(xy[0] * w))
    y = int(round(xy[1] * h))
    scale = min(h, w) / BASE_REF_SIZE
    crop_size = max(24, int(round(72 * scale)))
    search_pad = max(4, int(round(12 * scale)))
    return (x, y), crop_size, search_pad


def get_ruins_top_left_params(img: np.ndarray, center_xy_norm):
    (cx, cy), crop_size, search_pad = get_scaled_params(img, center_xy_norm)
    x = int(round(cx - 0.5 * crop_size))
    y = int(round(cy - 0.5 * crop_size))
    return (x, y), crop_size, search_pad


def ruins_status_subcrop(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape[:2]
    side = max(12, int(round(min(h, w) * 0.50)))
    x0 = max(0, w - side)
    y0 = max(0, h - side)
    return patch[y0:y0 + side, x0:x0 + side]


def crop_with_padding(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
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


def extract_templates(ref1: np.ndarray, ref2: np.ndarray):
    templates = {"top_left": {}, "bottom_right": {}}
    meta = {}
    ref2 = resize_to_match(ref1, ref2)
    for pos, cfg in CLASS_MAP.items():
        if pos in {"ruins", "great_church_top_right"}:
            continue
        (x, y), crop_size, search_pad = get_scaled_params(ref1, cfg["xy_norm"])
        templates[pos][REF1_LABELS[pos]] = crop_with_padding(ref1, x, y, crop_size)
        templates[pos][REF2_LABELS[pos]] = crop_with_padding(ref2, x, y, crop_size)
        meta[pos] = {"x": x, "y": y, "crop_size": crop_size, "search_pad": search_pad}

    # Ruins status templates are inferred from the two references.
    # Choose the most likely Ruins anchor among known center candidates.
    best_ruins = None
    for ruins_xy in RUINS_CANDIDATE_XY_NORMS:
        (rx, ry), rcrop, rpad = get_ruins_top_left_params(ref1, ruins_xy)
        p = crop_with_padding(ref1, rx, ry, rcrop)
        hsv = cv2.cvtColor(ruins_status_subcrop(p), cv2.COLOR_BGR2HSV)
        sat_strength = float(hsv[..., 1].mean())
        if best_ruins is None or sat_strength > best_ruins["sat_strength"]:
            best_ruins = {
                "xy": ruins_xy,
                "x": rx,
                "y": ry,
                "crop_size": rcrop,
                "search_pad": rpad,
                "sat_strength": sat_strength,
            }

    x = int(best_ruins["x"])
    y = int(best_ruins["y"])
    crop_size = int(best_ruins["crop_size"])
    search_pad = int(best_ruins["search_pad"])
    ref1_patch = crop_with_padding(ref1, x, y, crop_size)
    ref2_patch = crop_with_padding(ref2, x, y, crop_size)

    def _load_first_existing(paths):
        for p in paths:
            if Path(p).exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        return None

    poison_icon = _load_first_existing(POISON_ICON_CANDIDATES)
    bleed_icon = _load_first_existing(BLEED_ICON_CANDIDATES)
    lightning_icon = _load_first_existing(LIGHTNING_ICON_CANDIDATES)
    sleep_icon = _load_first_existing(SLEEP_ICON_CANDIDATES)

    ruins_probe = ruins_status_subcrop(ref1_patch)
    if poison_icon is not None and bleed_icon is not None:
        poison_tpl = cv2.resize(poison_icon, (ruins_probe.shape[1], ruins_probe.shape[0]), interpolation=cv2.INTER_AREA)
        bleed_tpl = cv2.resize(bleed_icon, (ruins_probe.shape[1], ruins_probe.shape[0]), interpolation=cv2.INTER_AREA)
        templates["ruins"] = {
            "Poison": poison_tpl,
            "Bleed": bleed_tpl,
        }
    else:
        # Fallback: if status icon files are missing, keep a two-template setup from refs.
        templates["ruins"] = {
            "Poison": ruins_status_subcrop(ref1_patch),
            "Bleed": ruins_status_subcrop(ref2_patch),
        }
    meta["ruins"] = {
        "x": int(x),
        "y": int(y),
        "crop_size": int(crop_size),
        "search_pad": int(search_pad),
        "candidate_xy_norms": [tuple(float(v) for v in xy) for xy in RUINS_CANDIDATE_XY_NORMS],
        "reference_center_xy_norm": tuple(float(v) for v in best_ruins["xy"]),
    }

    # Great Church (top-right) status templates: Lightning vs Sleep.
    # Use candidate-anchor probing (same robustness strategy as ruins).
    best_gc = None
    for gc_xy in GREAT_CHURCH_CANDIDATE_XY_NORMS:
        (gx, gy), gcrop, gpad = get_ruins_top_left_params(ref1, gc_xy)
        p = crop_with_padding(ref1, gx, gy, gcrop)
        hsv = cv2.cvtColor(ruins_status_subcrop(p), cv2.COLOR_BGR2HSV)
        sat_strength = float(hsv[..., 1].mean())
        if best_gc is None or sat_strength > best_gc["sat_strength"]:
            best_gc = {
                "xy": gc_xy,
                "x": gx,
                "y": gy,
                "crop_size": gcrop,
                "search_pad": gpad,
                "sat_strength": sat_strength,
            }

    gcx = int(best_gc["x"])
    gcy = int(best_gc["y"])
    gccrop = int(best_gc["crop_size"])
    gcpad = int(best_gc["search_pad"])
    gc_ref_patch = crop_with_padding(ref1, gcx, gcy, gccrop)
    gc_probe = ruins_status_subcrop(gc_ref_patch)
    if lightning_icon is not None and sleep_icon is not None:
        lightning_tpl = cv2.resize(lightning_icon, (gc_probe.shape[1], gc_probe.shape[0]), interpolation=cv2.INTER_AREA)
        sleep_tpl = cv2.resize(sleep_icon, (gc_probe.shape[1], gc_probe.shape[0]), interpolation=cv2.INTER_AREA)
        templates["great_church_top_right"] = {
            "Lightning": lightning_tpl,
            "Sleep": sleep_tpl,
        }
    else:
        gc_ref2_patch = crop_with_padding(ref2, gcx, gcy, gccrop)
        templates["great_church_top_right"] = {
            "Lightning": ruins_status_subcrop(gc_ref_patch),
            "Sleep": ruins_status_subcrop(gc_ref2_patch),
        }
    meta["great_church_top_right"] = {
        "x": int(gcx),
        "y": int(gcy),
        "crop_size": int(gccrop),
        "search_pad": int(gcpad),
        "candidate_xy_norms": [tuple(float(v) for v in xy) for xy in GREAT_CHURCH_CANDIDATE_XY_NORMS],
        "center_xy_norm": {
            "x_norm": float(best_gc["xy"][0]),
            "y_norm": float(best_gc["xy"][1]),
        },
    }
    return templates, meta


def feature_vector(img: np.ndarray):
    if img is None or img.size == 0:
        raise ValueError("feature_vector received an empty image patch")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2.0, W / 2.0
    radial = np.exp(-((((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5) / (min(H, W) / 2.6)) ** 2)

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


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2 * np.pi - d) / np.pi


def patch_similarity(a: np.ndarray, b: np.ndarray) -> float:
    fa = feature_vector(a)
    fb = feature_vector(b)
    hue_score = 1.0 - circular_distance(float(fa["hue"]), float(fb["hue"]))

    hist_score = float(cv2.compareHist(np.asarray(fa["color_hist"], dtype=np.float32), np.asarray(fb["color_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["color_hist"]).sum() > 0 and np.asarray(fb["color_hist"]).sum() > 0 else 0.0
    grad_score = float(cv2.compareHist(np.asarray(fa["grad_hist"], dtype=np.float32), np.asarray(fb["grad_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["grad_hist"]).sum() > 0 and np.asarray(fb["grad_hist"]).sum() > 0 else 0.0
    sv_score = 1.0 - (abs(float(fa["sat"]) - float(fb["sat"])) + abs(float(fa["val"]) - float(fb["val"]))) / 2.0
    return float(0.45 * hue_score + 0.30 * hist_score + 0.15 * grad_score + 0.10 * sv_score)


def classify_in_search_window(search: np.ndarray, templates):
    if search is None or search.size == 0:
        raise RuntimeError("Search window is empty after alignment.")

    template_h, template_w = next(iter(templates.values())).shape[:2]
    if template_h == 0 or template_w == 0:
        raise RuntimeError("Template crop is empty. Check that the reference images are valid.")
    if search.shape[0] < template_h or search.shape[1] < template_w:
        raise RuntimeError(f"Search window {search.shape[1]}x{search.shape[0]} is smaller than template {template_w}x{template_h}.")

    results = []
    for label, template in templates.items():
        best_score = -1.0
        best_loc = (0, 0)
        for dy in range(0, search.shape[0] - template_h + 1):
            for dx in range(0, search.shape[1] - template_w + 1):
                patch = search[dy:dy + template_h, dx:dx + template_w]
                score = patch_similarity(patch, template)
                if score > best_score:
                    best_score = score
                    best_loc = (dx, dy)
        results.append((label, float(best_score), best_loc))

    results.sort(key=lambda row: row[1], reverse=True)
    return results, (template_w, template_h)


def confidence_flag(margin: float) -> str:
    if margin >= 0.15:
        return "strong"
    if margin >= 0.08:
        return "medium"
    return "weak"


def inverse_transform_to_original(H: np.ndarray, original_query: np.ndarray, resized_query: np.ndarray) -> np.ndarray:
    H_inv = np.linalg.inv(H)
    oh, ow = original_query.shape[:2]
    rh, rw = resized_query.shape[:2]
    sx = rw / float(ow)
    sy = rh / float(oh)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.linalg.inv(S) @ H_inv


def project_box_to_original(box, ref_to_original: np.ndarray) -> np.ndarray:
    x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
    pts = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.float32)
    return np.round(cv2.perspectiveTransform(pts, ref_to_original)[0]).astype(np.int32)


def draw_text_label(img: np.ndarray, text: str, anchor):
    h, w = img.shape[:2]
    x = int(np.clip(anchor[0], 0, max(0, w - 1)))
    y = int(np.clip(anchor[1], 0, max(0, h - 1)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 1800.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    tx = min(max(4, x), max(4, w - tw - 6))
    ty = y - 8
    if ty - th - baseline < 0:
        ty = min(h - 6, y + th + 10)
    ty = max(th + baseline + 4, min(h - 4, ty))

    bg_x0, bg_y0 = max(0, tx - 3), max(0, ty - th - baseline - 3)
    bg_x1, bg_y1 = min(w, tx + tw + 3), min(h, ty + 3)
    cv2.rectangle(img, (bg_x0, bg_y0), (bg_x1, bg_y1), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)


def recognize_markers(reference1_path: str, reference2_path: str, query_path: str, save_debug: str | None = None, save_annotated: str | None = None) -> dict:
    ref1 = read_image(reference1_path)
    ref2 = read_image(reference2_path)
    query = read_image(query_path)

    aligned, H, original_query, resized_query = align_image(ref1, query)
    templates, meta = extract_templates(ref1, ref2)
    debug_img = aligned.copy()
    annotated_original = original_query.copy()
    ref_to_original = inverse_transform_to_original(H, original_query, resized_query)

    output = {"top_left": {}, "bottom_right": {}, "ruins": {}, "great_church_top_right": {}}
    for pos in ["top_left", "bottom_right", "ruins", "great_church_top_right"]:
        x, y = meta[pos]["x"], meta[pos]["y"]
        crop_size = meta[pos]["crop_size"]
        search_pad = meta[pos]["search_pad"]
        if pos in {"ruins", "great_church_top_right"}:
            best = None
            best_anchor = None
            for xy in meta[pos].get("candidate_xy_norms", []):
                (rx, ry), rcrop, rpad = get_ruins_top_left_params(aligned, xy)
                rsize = rcrop + 2 * rpad
                rsearch = crop_with_padding(aligned, rx - rpad, ry - rpad, rsize)
                ranked, (w, h) = classify_in_search_window(rsearch, templates[pos])
                winner, winner_score, loc = ranked[0]
                runner_up_score = ranked[1][1]
                confidence_margin = winner_score - runner_up_score
                dx, dy = loc
                abs_x = rx - rpad + dx
                abs_y = ry - rpad + dy
                candidate = {
                    "ranked": ranked,
                    "w": w,
                    "h": h,
                    "winner": winner,
                    "winner_score": winner_score,
                    "runner_up_score": runner_up_score,
                    "confidence_margin": confidence_margin,
                    "abs_x": abs_x,
                    "abs_y": abs_y,
                    "ref_x": rx,
                    "ref_y": ry,
                    "ref_crop_size": rcrop,
                    "ref_search_pad": rpad,
                }
                if best is None or float(candidate["confidence_margin"]) > float(best["confidence_margin"]):
                    best = candidate
                    best_anchor = xy
            ranked = best["ranked"]
            w, h = int(best["w"]), int(best["h"])
            winner = best["winner"]
            winner_score = float(best["winner_score"])
            runner_up_score = float(best["runner_up_score"])
            confidence_margin = float(best["confidence_margin"])
            abs_x = int(best["abs_x"])
            abs_y = int(best["abs_y"])
            x = int(best["ref_x"])
            y = int(best["ref_y"])
            crop_size = int(best["ref_crop_size"])
            search_pad = int(best["ref_search_pad"])
        else:
            search_size = crop_size + 2 * search_pad
            search = crop_with_padding(aligned, x - search_pad, y - search_pad, search_size)
            ranked, (w, h) = classify_in_search_window(search, templates[pos])
            winner, winner_score, loc = ranked[0]
            runner_up_score = ranked[1][1]
            confidence_margin = winner_score - runner_up_score
            dx, dy = loc
            abs_x = x - search_pad + dx
            abs_y = y - search_pad + dy
        box = {"x": int(abs_x), "y": int(abs_y), "w": int(w), "h": int(h)}
        if pos == "ruins":
            # User-calibrated positional correction on current GH panel crop path.
            box["x"] = int(box["x"] - 10)
        if pos == "great_church_top_right":
            # User-calibrated positional correction on current GH panel crop path.
            box["x"] = int(box["x"] - 30)
            box["y"] = int(box["y"] + 40)
        conf = confidence_flag(float(confidence_margin))

        output[pos] = {
            "label": winner,
            "score": round(float(winner_score), 4),
            "runner_up": ranked[1][0],
            "runner_up_score": round(float(runner_up_score), 4),
            "margin": round(float(confidence_margin), 4),
            "confidence": conf,
            "box": box,
            "reference_anchor": {"x": int(x), "y": int(y), "crop_size": int(crop_size), "search_pad": int(search_pad)},
            "ranked": [{"label": label, "score": round(float(score), 4), "dx": int(dx0), "dy": int(dy0)} for label, score, (dx0, dy0) in ranked],
        }
        if pos in {"ruins", "great_church_top_right"}:
            output[pos]["selected_candidate_xy_norm"] = {
                "x_norm": float(best_anchor[0]),
                "y_norm": float(best_anchor[1]),
            }
        if pos == "great_church_top_right":
            output[pos]["center_xy_norm"] = {
                "x_norm": float(meta[pos]["center_xy_norm"]["x_norm"]),
                "y_norm": float(meta[pos]["center_xy_norm"]["y_norm"]),
            }

        cv2.rectangle(debug_img, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 0, 255), 2)
        cv2.putText(debug_img, f"{pos}: {winner}", (abs_x, max(18, abs_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        poly = project_box_to_original(box, ref_to_original)
        cv2.polylines(annotated_original, [poly], True, (0, 0, 255), 2, cv2.LINE_AA)
        draw_text_label(annotated_original, f"{pos}: {winner} [{conf}]", tuple(poly[0]))

    if save_debug:
        debug_dir = Path(save_debug)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "aligned_query.png"), aligned)
        cv2.imwrite(str(debug_dir / "recognized_markers.png"), debug_img)

    if save_annotated:
        annotated_path = Path(save_annotated)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated_original)

    return output


def iter_images(folder: Path, recursive: bool = False):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def run_folder_mode(reference1: str, reference2: str, input_folder: str, output_csv: str, output_json: str | None, save_debug: str | None, save_annotated: str | None, recursive: bool) -> None:
    import csv

    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a folder: {input_folder}")

    images = list(iter_images(folder, recursive=recursive))
    if not images:
        raise SystemExit("No image files found in the input folder.")

    rows, detailed = [], []
    for img_path in images:
        try:
            debug_dir = str(Path(save_debug) / img_path.relative_to(folder).with_suffix("")) if save_debug else None
            annotated_path = str(Path(save_annotated) / img_path.relative_to(folder)) if save_annotated else None
            result = recognize_markers(reference1, reference2, str(img_path), save_debug=debug_dir, save_annotated=annotated_path)
            tl, br = result["top_left"], result["bottom_right"]
            ru = result.get("ruins", {})
            gc = result.get("great_church_top_right", {})
            row = {
                "file": str(img_path.relative_to(folder)),
                "top_left": tl["label"],
                "top_left_score": tl["score"],
                "top_left_margin": tl["margin"],
                "top_left_confidence": tl["confidence"],
                "bottom_right": br["label"],
                "bottom_right_score": br["score"],
                "bottom_right_margin": br["margin"],
                "bottom_right_confidence": br["confidence"],
                "ruins": ru.get("label", ""),
                "ruins_score": ru.get("score", ""),
                "ruins_margin": ru.get("margin", ""),
                "ruins_confidence": ru.get("confidence", ""),
                "great_church_top_right": gc.get("label", ""),
                "great_church_top_right_score": gc.get("score", ""),
                "great_church_top_right_margin": gc.get("margin", ""),
                "great_church_top_right_confidence": gc.get("confidence", ""),
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "ok", "result": result})
            print(
                f"[OK] {row['file']} -> top_left={row['top_left']} ({row['top_left_confidence']}), "
                f"bottom_right={row['bottom_right']} ({row['bottom_right_confidence']}), "
                f"ruins={row['ruins']} ({row['ruins_confidence']}), "
                f"great_church_top_right={row['great_church_top_right']} ({row['great_church_top_right_confidence']})"
            )
        except Exception as exc:
            row = {
                "file": str(img_path.relative_to(folder)),
                "top_left": "",
                "top_left_score": "",
                "top_left_margin": "",
                "top_left_confidence": "",
                "bottom_right": "",
                "bottom_right_score": "",
                "bottom_right_margin": "",
                "bottom_right_confidence": "",
                "ruins": "",
                "ruins_score": "",
                "ruins_margin": "",
                "ruins_confidence": "",
                "great_church_top_right": "",
                "great_church_top_right_score": "",
                "great_church_top_right_margin": "",
                "great_church_top_right_confidence": "",
                "status": "error",
                "error": str(exc),
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "error", "error": str(exc)})
            print(f"[ERROR] {row['file']} -> {exc}")

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "top_left",
                "top_left_score",
                "top_left_margin",
                "top_left_confidence",
                "bottom_right",
                "bottom_right_score",
                "bottom_right_margin",
                "bottom_right_confidence",
                "ruins",
                "ruins_score",
                "ruins_margin",
                "ruins_confidence",
                "great_church_top_right",
                "great_church_top_right_score",
                "great_church_top_right_margin",
                "great_church_top_right_confidence",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2)

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    err_count = len(rows) - ok_count
    print(f"\nDone. Processed {len(rows)} image(s): {ok_count} ok, {err_count} error(s).")
    print(f"CSV written to: {output_csv_path}")
    if output_json:
        print(f"JSON written to: {output_json}")
    if save_debug:
        print(f"Debug output root: {save_debug}")
    if save_annotated:
        print(f"Annotated image root: {save_annotated}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize the two square marker classes on one image or a whole folder.")
    parser.add_argument("reference1", help="Reference image where top-left=Fire and bottom-right=Rot")
    parser.add_argument("reference2", help="Reference image where top-left=Magic and bottom-right=Holy")
    parser.add_argument("input_path", help="Image file to classify, or a folder of images")
    parser.add_argument("--save-debug", default=None, help="Optional folder for aligned-image debug output")
    parser.add_argument("--save-annotated", default=None, help="Save annotated copies of the original query image. For a folder input, pass a folder. For a single image, pass an output image path.")
    parser.add_argument("--output-csv", default="recognizer_results.csv", help="In folder mode, write summary CSV here")
    parser.add_argument("--output-json", default=None, help="In folder mode, optionally write detailed JSON here")
    parser.add_argument("--recursive", action="store_true", help="In folder mode, also scan subfolders")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if input_path.is_dir():
        run_folder_mode(args.reference1, args.reference2, str(input_path), args.output_csv, args.output_json, args.save_debug, args.save_annotated, args.recursive)
    else:
        result = recognize_markers(args.reference1, args.reference2, str(input_path), save_debug=args.save_debug, save_annotated=args.save_annotated)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
