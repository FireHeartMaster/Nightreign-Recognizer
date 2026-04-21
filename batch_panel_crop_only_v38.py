#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Iterable, List

import cv2

import nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay as v38


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch panel-crop full screenshot images using v38 circle detection. "
            "This script only performs panel crop (no further crop/normalization)."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing full screenshot images.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory where cropped panel images will be saved.")
    p.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--suffix",
        type=str,
        default="_panel",
        help="Suffix appended to output filename stem. Default: _panel",
    )
    p.add_argument("--verbose", action="store_true", help="Enable v38 internal crop-stage logs per image.")
    return p.parse_args()


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        it = root.rglob("*")
    else:
        it = root.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def make_run_context(verbose: bool) -> v38.RunContext:
    def _log(_run_id: int, msg: str) -> None:
        if verbose:
            print(msg)

    return v38.RunContext(run_id=1, cancel_event=threading.Event(), log_func=_log)


def crop_panel_only(full_img_bgr, ctx: v38.RunContext):
    circle, crop_method = v38.detect_map_circle(
        full_img_bgr,
        v38.DEFAULT_HOUGH_REGION,
        v38.DEFAULT_HOUGH_SCALE,
        ctx,
    )
    if crop_method == "blue_ring":
        crop_box = v38.clip_box(
            v38.estimate_panel_box(circle, v38.DEFAULT_PANEL_SCALE),
            full_img_bgr.shape[1],
            full_img_bgr.shape[0],
        )
    else:
        crop_box = v38.refine_panel_box(
            full_img_bgr,
            circle,
            v38.DEFAULT_PANEL_SCALE,
            v38.DEFAULT_SEARCH_RADIUS,
            v38.DEFAULT_PAD,
            ctx,
        )
    panel = full_img_bgr[crop_box.y0 : crop_box.y1, crop_box.x0 : crop_box.x1].copy()
    return panel, crop_box, crop_method


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = list(iter_images(input_dir, bool(args.recursive)))
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    print(f"Found {len(images)} images.")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("Mode: panel crop only (no further crop/normalization)")

    success = 0
    failed = 0

    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(input_dir)
        out_rel = rel.with_name(f"{rel.stem}{args.suffix}{rel.suffix}")
        out_path = output_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            print(f"[{idx:04d}/{len(images):04d}] skip (exists): {out_rel}")
            continue

        full_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if full_img is None:
            failed += 1
            print(f"[{idx:04d}/{len(images):04d}] fail (read): {rel}")
            continue

        try:
            ctx = make_run_context(bool(args.verbose))
            panel, crop_box, crop_method = crop_panel_only(full_img, ctx)
            ok = cv2.imwrite(str(out_path), panel)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {out_path}")
            success += 1
            print(
                f"[{idx:04d}/{len(images):04d}] ok {out_rel} "
                f"method={crop_method} box=({crop_box.x0},{crop_box.y0})-({crop_box.x1},{crop_box.y1}) "
                f"size={panel.shape[1]}x{panel.shape[0]}"
            )
        except Exception as exc:
            failed += 1
            print(f"[{idx:04d}/{len(images):04d}] fail (crop): {rel} :: {exc}")

    print("")
    print("Done.")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
