import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE_REF_SIZE = 2048.0
TOP_LEFT_XY_NORM = (635.0 / BASE_REF_SIZE, 538.0 / BASE_REF_SIZE)
BOTTOM_RIGHT_XY_NORM = (1363.0 / BASE_REF_SIZE, 1400.0 / BASE_REF_SIZE)

CLASS_MAP = {
    "top_left": {"allowed": ["Fire", "Magic"], "xy_norm": TOP_LEFT_XY_NORM},
    "bottom_right": {"allowed": ["Rot", "Holy"], "xy_norm": BOTTOM_RIGHT_XY_NORM},
}

REF1_LABELS = {"top_left": "Fire", "bottom_right": "Rot"}
REF2_LABELS = {"top_left": "Magic", "bottom_right": "Holy"}


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
    x = int(round(xy_norm[0] * w))
    y = int(round(xy_norm[1] * h))
    scale = min(h, w) / BASE_REF_SIZE
    crop_size = max(24, int(round(72 * scale)))
    search_pad = max(4, int(round(12 * scale)))
    return (x, y), crop_size, search_pad


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
        (x, y), crop_size, search_pad = get_scaled_params(ref1, cfg["xy_norm"])
        templates[pos][REF1_LABELS[pos]] = crop_with_padding(ref1, x, y, crop_size)
        templates[pos][REF2_LABELS[pos]] = crop_with_padding(ref2, x, y, crop_size)
        meta[pos] = {"x": x, "y": y, "crop_size": crop_size, "search_pad": search_pad}
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

    output = {"top_left": {}, "bottom_right": {}}
    for pos in ["top_left", "bottom_right"]:
        x, y = meta[pos]["x"], meta[pos]["y"]
        crop_size = meta[pos]["crop_size"]
        search_pad = meta[pos]["search_pad"]
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
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "ok", "result": result})
            print(f"[OK] {row['file']} -> top_left={row['top_left']} ({row['top_left_confidence']}), bottom_right={row['bottom_right']} ({row['bottom_right_confidence']})")
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
                "status": "error",
                "error": str(exc),
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "error", "error": str(exc)})
            print(f"[ERROR] {row['file']} -> {exc}")

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "top_left", "top_left_score", "top_left_margin", "top_left_confidence", "bottom_right", "bottom_right_score", "bottom_right_margin", "bottom_right_confidence", "status", "error"])
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
