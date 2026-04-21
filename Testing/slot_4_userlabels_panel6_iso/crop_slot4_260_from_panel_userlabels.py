import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2


# Slot-4 reference from prior pipeline test summaries:
# slot4_anchor_reference = {x_asym_px_tl: 809, y_asym_px_tl: 625, crop_side: 200}
# Convert asym top-left back to 1536 panel by adding margins:
# x_panel_tl_200 = 809 + 175 = 984 ; y_panel_tl_200 = 625 + 180 = 805
SLOT4_CROP200_TL_X_PANEL = 984
SLOT4_CROP200_TL_Y_PANEL = 805
SLOT4_CROP200_SIDE = 200
BASE_SIZE = 260


def clamp_rect(x0: int, y0: int, size: int, w: int, h: int):
    x0 = max(0, min(w - size, x0))
    y0 = max(0, min(h - size, y0))
    return x0, y0


def crop_top_left(img, x0: int, y0: int, size: int):
    h, w = img.shape[:2]
    x0, y0 = clamp_rect(x0, y0, size, w, h)
    return img[y0 : y0 + size, x0 : x0 + size].copy(), x0, y0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slot-4 260x260 base crops from user-labeled panel crops.")
    parser.add_argument(
        "--panel-dir",
        default="tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric/Screenshots/panel_crop",
    )
    parser.add_argument("--out-dir", default="Testing/slot_4_userlabels_panel6_iso/data/bases260")
    parser.add_argument("--manifest", default="Testing/slot_4_userlabels_panel6_iso/data/bases260_manifest.json")
    args = parser.parse_args()

    panel_dir = Path(args.panel_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # User-provided labels.
    labels: Dict[str, str] = {
        "1115": "Sorcerer_s_Rise",
        "1119": "Church",
        "1120": "Church",
        "1128": "empty",
        "1175": "empty",
        "1185": "Church",
    }

    items: List[dict] = []
    for layout_id, label in labels.items():
        src = panel_dir / f"{layout_id}_panel_crop.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing panel crop: {src}")
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {src}")

        h, w = img.shape[:2]
        x200_tl = int(round((SLOT4_CROP200_TL_X_PANEL / 1536.0) * w))
        y200_tl = int(round((SLOT4_CROP200_TL_Y_PANEL / 1536.0) * h))
        x200_tl, y200_tl = clamp_rect(x200_tl, y200_tl, SLOT4_CROP200_SIDE, w, h)

        cx = x200_tl + (SLOT4_CROP200_SIDE // 2)
        cy = y200_tl + (SLOT4_CROP200_SIDE // 2)
        x260_tl = cx - (BASE_SIZE // 2)
        y260_tl = cy - (BASE_SIZE // 2)
        crop, x0, y0 = crop_top_left(img, x260_tl, y260_tl, BASE_SIZE)

        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"pattern_{layout_id}__slot4_panel260.png"
        out_path = class_dir / out_name
        cv2.imwrite(str(out_path), crop)

        items.append(
            {
                "layout_id": int(layout_id),
                "label": label,
                "source_path": str(src),
                "output_path": str(out_path),
                "panel_shape": {"w": int(w), "h": int(h)},
                "crop200_ref_tl_px": {"x": int(x200_tl), "y": int(y200_tl), "size": int(SLOT4_CROP200_SIDE)},
                "crop260_rect": {"x0": int(x0), "y0": int(y0), "size": int(BASE_SIZE)},
            }
        )

    manifest = {
        "notes": "Isolated slot-4 dataset from user-labeled panel crops only.",
        "slot4_crop200_tl_panel_ref_1536": {
            "x": SLOT4_CROP200_TL_X_PANEL,
            "y": SLOT4_CROP200_TL_Y_PANEL,
            "size": SLOT4_CROP200_SIDE,
        },
        "base_size": BASE_SIZE,
        "items": items,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} base crops to: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

