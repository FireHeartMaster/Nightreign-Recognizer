#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2

import nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay as v38


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Normalize panel-crop images in batch using v38 slot normalization "
            "(same function used by the v38 recognition pipeline)."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing panel-crop images.")
    p.add_argument("--output-dir", type=Path, required=True, help="Folder where normalized images are saved.")
    p.add_argument("--recursive", action="store_true", help="Scan input folder recursively.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist.")
    p.add_argument(
        "--suffix",
        type=str,
        default="_norm",
        help="Suffix appended to output filename stem. Default: _norm",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=int(v38.SLOT_CANONICAL_PANEL_SIZE),
        help=f"Output square size. Default: {int(v38.SLOT_CANONICAL_PANEL_SIZE)}",
    )
    return p.parse_args()


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    target_size = max(32, int(args.target_size))

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir, bool(args.recursive)))
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    print(f"Found {len(images)} images.")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target normalization size: {target_size}")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, src in enumerate(images, start=1):
        rel = src.relative_to(input_dir)
        out_rel = rel.with_name(f"{rel.stem}{args.suffix}{rel.suffix}")
        out_path = output_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skip_count += 1
            print(f"[{idx:04d}/{len(images):04d}] skip (exists): {out_rel}")
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            fail_count += 1
            print(f"[{idx:04d}/{len(images):04d}] fail (read): {rel}")
            continue

        try:
            norm = v38.normalize_panel_for_slot_recognition(img, target_size=target_size)
            if norm is None or norm.size == 0:
                raise RuntimeError("Normalization returned empty image.")
            if not cv2.imwrite(str(out_path), norm):
                raise RuntimeError("cv2.imwrite failed.")
            ok_count += 1
            print(
                f"[{idx:04d}/{len(images):04d}] ok {out_rel} "
                f"{img.shape[1]}x{img.shape[0]} -> {norm.shape[1]}x{norm.shape[0]}"
            )
        except Exception as exc:
            fail_count += 1
            print(f"[{idx:04d}/{len(images):04d}] fail (normalize): {rel} :: {exc}")

    print("")
    print("Done.")
    print(f"Success: {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed:  {fail_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

