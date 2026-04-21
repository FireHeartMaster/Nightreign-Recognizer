import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


BASE_SIZE = 260
OUT_SIZE = 200
HALF_OUT = OUT_SIZE // 2
CENTER = BASE_SIZE // 2


def build_affine_matrix(
    angle_deg: float,
    scale: float,
    shear_deg: float,
    tx_px: float,
    ty_px: float,
) -> np.ndarray:
    a = math.radians(angle_deg)
    sh = math.radians(shear_deg)

    rot = np.array(
        [
            [math.cos(a), -math.sin(a)],
            [math.sin(a), math.cos(a)],
        ],
        dtype=np.float32,
    )
    sc = np.array([[scale, 0.0], [0.0, scale]], dtype=np.float32)
    shear = np.array([[1.0, math.tan(sh)], [0.0, 1.0]], dtype=np.float32)
    linear = rot @ shear @ sc

    c = np.array([CENTER, CENTER], dtype=np.float32)
    t = np.array([tx_px, ty_px], dtype=np.float32) + c - (linear @ c)
    m = np.concatenate([linear, t.reshape(2, 1)], axis=1)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slot-2 200x200 jitters with affine+reflect from 260x260 bases.")
    parser.add_argument("--bases-dir", default="Testing/slot_2_fixed_geometry_panel_bases260/data/bases260")
    parser.add_argument("--out-dir", default="Testing/slot_2_fixed_geometry_panel_bases260/data/jitters200_3000_per_class")
    parser.add_argument(
        "--manifest",
        default="Testing/slot_2_fixed_geometry_panel_bases260/data/jitters200_3000_per_class_manifest.json",
    )
    parser.add_argument("--per-class", type=int, default=3000, help="Target jitter count per class.")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    bases_dir = Path(args.bases_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in sorted(bases_dir.iterdir()) if p.is_dir()]
    summary: Dict[str, Dict[str, object]] = {}
    items: List[dict] = []

    for class_dir in class_dirs:
        label = class_dir.name
        base_files = sorted(
            [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
        )
        if not base_files:
            continue

        target_dir = out_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        n_bases = len(base_files)
        total = int(args.per_class)
        base_quota = total // n_bases
        remainder = total % n_bases
        quotas = [base_quota + (1 if i < remainder else 0) for i in range(n_bases)]

        class_count = 0
        per_base_counts: Dict[str, int] = {}
        for i, base_path in enumerate(base_files):
            img = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read base image: {base_path}")
            if img.shape[0] != BASE_SIZE or img.shape[1] != BASE_SIZE:
                raise RuntimeError(f"Base must be {BASE_SIZE}x{BASE_SIZE}: {base_path} got {img.shape[:2]}")

            n_samples = quotas[i]
            per_base_counts[base_path.name] = int(n_samples)
            stem = base_path.stem
            for j in range(n_samples):
                angle = float(rng.uniform(-8.0, 8.0))
                scale = float(rng.uniform(0.92, 1.08))
                shear = float(rng.uniform(-5.0, 5.0))
                tx_aff = float(rng.uniform(-6.0, 6.0))
                ty_aff = float(rng.uniform(-6.0, 6.0))
                m = build_affine_matrix(
                    angle_deg=angle,
                    scale=scale,
                    shear_deg=shear,
                    tx_px=tx_aff,
                    ty_px=ty_aff,
                )
                warped = cv2.warpAffine(
                    img,
                    m,
                    (BASE_SIZE, BASE_SIZE),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

                dx = int(rng.integers(-30, 31))
                dy = int(rng.integers(-30, 31))
                cx = CENTER + dx
                cy = CENTER + dy
                x0 = cx - HALF_OUT
                y0 = cy - HALF_OUT
                patch = warped[y0 : y0 + OUT_SIZE, x0 : x0 + OUT_SIZE].copy()

                alpha = float(rng.uniform(0.92, 1.08))
                beta = float(rng.uniform(-10.0, 10.0))
                patch = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)

                out_name = f"{stem}__j{j+1:04d}__dx{dx:+d}_dy{dy:+d}.png"
                out_path = target_dir / out_name
                cv2.imwrite(str(out_path), patch)
                class_count += 1

            items.append(
                {
                    "label": label,
                    "base_image": str(base_path),
                    "generated": int(n_samples),
                }
            )

        summary[label] = {
            "total_generated": int(class_count),
            "base_images": int(n_bases),
            "per_base": per_base_counts,
        }

    payload = {
        "notes": "slot-2 jitter set; affine jitter + reflected borders; 260->200 crops.",
        "per_class_target": int(args.per_class),
        "seed": int(args.seed),
        "base_size": BASE_SIZE,
        "out_size": OUT_SIZE,
        "summary": summary,
        "items": items,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Generated jitters under: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
