#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

PANEL_REF = 1536.0
GH_ASYM_INPUT = 1536
GH_ASYM_CROP = {"left": 175, "right": 170, "top": 180, "bottom": 180}
CROP_SIZE = 260

# Centers used by GH parallel panel classifiers.
# panel_raw coordinates are in raw panel space (after panel crop).
# gh_asym coordinates are in asymmetric GH panel space.
TARGETS: Dict[str, Dict[str, float | str]] = {
    "slot2": {"space": "panel_raw_norm_dx", "x_norm": 0.82890625, "y_norm": 0.38046875, "dx_px": 50.0, "dy_px": 0.0},
    "slot4": {"space": "panel_raw_abs", "x": 984.0, "y": 805.0},
    "slot5": {"space": "panel_raw_abs", "x": 499.0, "y": 809.0},
    "slot7": {"space": "panel_raw_abs", "x": 514.0, "y": 1265.0},
    "slot6": {"space": "gh_asym_abs", "x": 144.0, "y": 731.0},
    "palace": {"space": "gh_asym_abs", "x": 329.0, "y": 251.0},
    "fortress": {"space": "gh_asym_abs", "x": 875.0, "y": 897.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract 260x260 crops for Great Hollow slot/palace/fortress targets "
            "from all images in a folder. Creates one output folder per target."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing input images.")
    p.add_argument("--output-dir", type=Path, required=True, help="Folder where target subfolders will be created.")
    p.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return p.parse_args()


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def preprocess_gh_asym(panel_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(panel_bgr, (GH_ASYM_INPUT, GH_ASYM_INPUT), interpolation=cv2.INTER_CUBIC)
    l = int(GH_ASYM_CROP["left"])
    r = int(GH_ASYM_CROP["right"])
    t = int(GH_ASYM_CROP["top"])
    b = int(GH_ASYM_CROP["bottom"])
    y0 = max(0, t)
    y1 = min(GH_ASYM_INPUT, GH_ASYM_INPUT - b)
    x0 = max(0, l)
    x1 = min(GH_ASYM_INPUT, GH_ASYM_INPUT - r)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid GH asym crop bounds.")
    return resized[y0:y1, x0:x1].copy()


def crop_with_padding(img_bgr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    half = int(size) // 2
    x0 = int(cx) - half
    y0 = int(cy) - half
    x1 = x0 + int(size)
    y1 = y0 + int(size)

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)
    patch = img_bgr[y0c:y1c, x0c:x1c]

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        patch = cv2.copyMakeBorder(
            patch,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return patch


def compute_target_center(panel_raw: np.ndarray, gh_asym: np.ndarray, target_cfg: Dict[str, float | str]) -> Tuple[np.ndarray, int, int]:
    space = str(target_cfg["space"])
    if space == "panel_raw_abs":
        src = panel_raw
        cx = int(round(float(target_cfg["x"])))
        cy = int(round(float(target_cfg["y"])))
        return src, cx, cy
    if space == "gh_asym_abs":
        src = gh_asym
        cx = int(round(float(target_cfg["x"])))
        cy = int(round(float(target_cfg["y"])))
        return src, cx, cy
    if space == "panel_raw_norm_dx":
        src = panel_raw
        h, w = src.shape[:2]
        cx = int(round(float(target_cfg["x_norm"]) * w + float(target_cfg["dx_px"])))
        cy = int(round(float(target_cfg["y_norm"]) * h + float(target_cfg["dy_px"])))
        return src, cx, cy
    raise ValueError(f"Unknown target space: {space}")


def safe_stem_from_relative(rel_path: Path) -> str:
    parent = str(rel_path.parent).replace("\\", "__").replace("/", "__").strip(".")
    if parent and parent != ".":
        return f"{parent}__{rel_path.stem}"
    return rel_path.stem


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        (output_dir / target).mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir, bool(args.recursive)))
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Output root: {output_dir}")
    print(f"Targets: {', '.join(TARGETS.keys())}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")

    ok = 0
    fail = 0
    skip = 0

    for idx, src_path in enumerate(images, start=1):
        rel = src_path.relative_to(input_dir)
        base_name = safe_stem_from_relative(rel)
        panel_raw = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if panel_raw is None:
            fail += 1
            print(f"[{idx:04d}/{len(images):04d}] fail(read): {rel}")
            continue

        try:
            gh_asym = preprocess_gh_asym(panel_raw)
        except Exception as exc:
            fail += 1
            print(f"[{idx:04d}/{len(images):04d}] fail(preprocess): {rel} :: {exc}")
            continue

        image_had_failure = False
        image_had_skip = False
        for target_name, cfg in TARGETS.items():
            out_path = output_dir / target_name / f"{base_name}.png"
            if out_path.exists() and not args.overwrite:
                image_had_skip = True
                continue
            try:
                src_img, cx, cy = compute_target_center(panel_raw, gh_asym, cfg)
                patch = crop_with_padding(src_img, cx, cy, CROP_SIZE)
                if patch.shape[0] != CROP_SIZE or patch.shape[1] != CROP_SIZE:
                    raise RuntimeError(f"Unexpected crop size {patch.shape[1]}x{patch.shape[0]}")
                if not cv2.imwrite(str(out_path), patch):
                    raise RuntimeError("cv2.imwrite failed")
            except Exception as exc:
                image_had_failure = True
                print(f"[{idx:04d}/{len(images):04d}] fail({target_name}): {rel} :: {exc}")

        if image_had_failure:
            fail += 1
        elif image_had_skip:
            skip += 1
            print(f"[{idx:04d}/{len(images):04d}] skip(partial exists): {rel}")
        else:
            ok += 1
            print(f"[{idx:04d}/{len(images):04d}] ok: {rel}")

    print("")
    print("Done.")
    print(f"Images succeeded: {ok}")
    print(f"Images skipped:   {skip}")
    print(f"Images failed:    {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

