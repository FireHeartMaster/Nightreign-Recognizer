#!/usr/bin/env python3
"""
Convert all PNG files in a folder to WebP.

Examples:
  python convert_pngs_to_webp.py ./icons
  python convert_pngs_to_webp.py ./icons --output-dir ./icons_webp
  python convert_pngs_to_webp.py ./icons --recursive
  python convert_pngs_to_webp.py ./icons --lossless
  python convert_pngs_to_webp.py ./icons --quality 90 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PNG files in a folder to WebP.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing PNG files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to write WebP files to. Default: <input_dir>_webp"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="WebP quality for lossy mode (0-100). Default: 95"
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use lossless WebP instead of lossy."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .webp files in the output directory."
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the original PNG after successful conversion."
    )
    return parser.parse_args()


def find_pngs(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    files = [p for p in input_dir.glob(pattern) if p.is_file()]
    # Also include uppercase .PNG files
    if recursive:
        files.extend([p for p in input_dir.glob("**/*.PNG") if p.is_file()])
    else:
        files.extend([p for p in input_dir.glob("*.PNG") if p.is_file()])
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def convert_one(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    lossless: bool,
    quality: int,
    overwrite: bool,
) -> tuple[bool, Path, str]:
    try:
        relative = src.relative_to(input_dir)
        dst = output_dir / relative.with_suffix(".webp")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            return False, dst, "skipped (already exists)"

        with Image.open(src) as im:
            # Preserve alpha if present
            if im.mode not in ("RGB", "RGBA", "LA", "L", "P"):
                im = im.convert("RGBA")
            elif im.mode == "P" and "transparency" in im.info:
                im = im.convert("RGBA")
            elif im.mode == "P":
                im = im.convert("RGB")

            save_kwargs = {"format": "WEBP"}
            if lossless:
                save_kwargs["lossless"] = True
                save_kwargs["quality"] = 100
            else:
                save_kwargs["quality"] = max(0, min(100, quality))
                save_kwargs["method"] = 6

            im.save(dst, **save_kwargs)

        return True, dst, "converted"
    except Exception as e:
        return False, src, f"error: {e}"


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else input_dir.parent / f"{input_dir.name}_webp"
    output_dir.mkdir(parents=True, exist_ok=True)

    pngs = find_pngs(input_dir, args.recursive)
    if not pngs:
        print("No PNG files found.")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for src in pngs:
        ok, path, status = convert_one(
            src=src,
            input_dir=input_dir,
            output_dir=output_dir,
            lossless=args.lossless,
            quality=args.quality,
            overwrite=args.overwrite,
        )

        if status.startswith("error:"):
            failed += 1
            print(f"[FAIL] {src} -> {status}", file=sys.stderr)
            continue

        if status.startswith("skipped"):
            skipped += 1
            print(f"[SKIP] {src} -> {path}")
            continue

        converted += 1
        print(f"[OK]   {src} -> {path}")

        if args.delete_source:
            try:
                src.unlink()
            except Exception as e:
                failed += 1
                print(f"[WARN] Converted but could not delete source {src}: {e}", file=sys.stderr)

    print()
    print(f"Done. Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
