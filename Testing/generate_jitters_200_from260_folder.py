import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


BASE_SIZE = 260
OUT_SIZE = 200
HALF_OUT = OUT_SIZE // 2
CENTER = BASE_SIZE // 2
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 200x200 translated crop jitters from 260x260 images (50 per image by default)."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with base images.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output folder. Default: <input-dir>/jitters_200_by_source",
    )
    parser.add_argument(
        "--per-image",
        type=int,
        default=50,
        help="How many jitters to generate per source image.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=30,
        help="Max absolute jitter in pixels for dx/dy (range is [-jitter, +jitter]).",
    )
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Path for JSON summary. Default: <output-dir>/jitter_from_260_folder_summary.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {input_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (input_dir / "jitters_200_by_source").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else (output_dir / "jitter_from_260_folder_summary.json").resolve()
    )

    per_image = int(args.per_image)
    jitter = int(args.jitter)
    if per_image <= 0:
        raise RuntimeError("--per-image must be > 0")
    if jitter < 0:
        raise RuntimeError("--jitter must be >= 0")

    rng = np.random.default_rng(int(args.seed))
    sources = _collect_images(input_dir)
    if not sources:
        raise RuntimeError(f"No images found in: {input_dir}")

    created_count = 0
    errors: List[Dict[str, str]] = []
    entries: List[Dict[str, object]] = []

    for src in sources:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            errors.append({"source": str(src), "error": "Could not read image"})
            continue

        h, w = img.shape[:2]
        if h != BASE_SIZE or w != BASE_SIZE:
            errors.append(
                {
                    "source": str(src),
                    "error": f"Expected {BASE_SIZE}x{BASE_SIZE}, got {w}x{h}",
                }
            )
            continue

        target_dir = output_dir / src.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        local_created = 0
        for j in range(per_image):
            dx = int(rng.integers(-jitter, jitter + 1))
            dy = int(rng.integers(-jitter, jitter + 1))

            cx = CENTER + dx
            cy = CENTER + dy
            x0 = cx - HALF_OUT
            y0 = cy - HALF_OUT
            crop = img[y0 : y0 + OUT_SIZE, x0 : x0 + OUT_SIZE].copy()

            out_name = f"{src.stem}__from260__j{j+1:03d}__dx{dx:+d}_dy{dy:+d}.png"
            out_path = target_dir / out_name
            cv2.imwrite(str(out_path), crop)
            local_created += 1
            created_count += 1

        entries.append(
            {
                "source": str(src),
                "source_shape": [int(h), int(w)],
                "target_subfolder": str(target_dir),
                "created": int(local_created),
            }
        )

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "crop_size": [OUT_SIZE, OUT_SIZE],
        "base_size_expected": [BASE_SIZE, BASE_SIZE],
        "jitter_range": [-jitter, jitter],
        "per_image_target": per_image,
        "seed": int(args.seed),
        "source_count": len(sources),
        "created_count": created_count,
        "error_count": len(errors),
        "sources": entries,
        "errors": errors,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Created crops: {created_count}")
    print(f"Errors: {len(errors)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
