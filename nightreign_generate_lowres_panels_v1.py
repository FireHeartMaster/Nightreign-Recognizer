#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate lower-resolution simulated map panels from already-cropped 1536x1536-style "
            "reference images. The script downsamples to a smaller panel size to mimic a low-resolution "
            "capture, optionally adds light blur/compression, then upsamples back to the original size."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing numbered panel images.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory where simulated panels will be written.")
    p.add_argument(
        "--target-panel-size",
        type=int,
        default=780,
        help=(
            "Intermediate square size used to simulate the lower-resolution panel before resizing back "
            "to the original size. 780 is a good first approximation for the earlier 1080p capture behavior."
        ),
    )
    p.add_argument("--blur-sigma", type=float, default=0.8, help="Optional Gaussian blur sigma applied after downscaling.")
    p.add_argument("--jpeg-quality", type=int, default=88, help="Optional JPEG recompression quality (1-100). Set <=0 to disable.")
    p.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply a mild sharpen after upscaling, to mimic some display pipelines.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def list_numbered_images(images_dir: Path):
    files = []
    for path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No numbered images found in {images_dir}")
    return files


def maybe_recompress(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    if quality <= 0:
        return img_bgr
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img_bgr


def mild_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 0.8)
    return cv2.addWeighted(img_bgr, 1.18, blur, -0.18, 0)


def simulate_lowres_panel(img_bgr: np.ndarray, target_panel_size: int, blur_sigma: float, jpeg_quality: int, sharpen: bool) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Empty image.")
    target_panel_size = max(64, int(target_panel_size))

    # Downscale to the effective low-resolution panel size.
    small = cv2.resize(img_bgr, (target_panel_size, target_panel_size), interpolation=cv2.INTER_AREA)

    if blur_sigma > 1e-6:
        small = cv2.GaussianBlur(small, (0, 0), blur_sigma)

    small = maybe_recompress(small, jpeg_quality)

    # Upscale back to original size, because the recognizer later normalizes to a large canonical panel.
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    if sharpen:
        up = mild_sharpen(up)

    return up


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list_numbered_images(args.input_dir)
    print(f"Found {len(files)} numbered images in {args.input_dir}")
    print(
        f"Generating simulated low-res panels with target_panel_size={args.target_panel_size}, "
        f"blur_sigma={args.blur_sigma}, jpeg_quality={args.jpeg_quality}, sharpen={args.sharpen}"
    )

    for idx, path in enumerate(files, start=1):
        out_path = args.output_dir / path.name
        if out_path.exists() and not args.overwrite:
            print(f"[{idx:03d}/{len(files)}] Skipping existing {out_path.name}")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{idx:03d}/{len(files)}] Failed to read {path}")
            continue
        sim = simulate_lowres_panel(
            img_bgr=img,
            target_panel_size=args.target_panel_size,
            blur_sigma=args.blur_sigma,
            jpeg_quality=args.jpeg_quality,
            sharpen=args.sharpen,
        )
        cv2.imwrite(str(out_path), sim)
        print(f"[{idx:03d}/{len(files)}] Wrote {out_path.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
