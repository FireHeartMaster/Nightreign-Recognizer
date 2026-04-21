import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SLOT2_CENTER_NORM_X = 0.82890625
SLOT2_CENTER_NORM_Y = 0.38046875
BASE_SIZE = 260
SLOT2_OFFSET_X_PX = 50
SLOT2_OFFSET_Y_PX = 0

STATUS_TO_CLASS = {
    "Status_Lightning": "Great_Church_Lightning",
    "Status_Sleep": "Great_Church_Sleep",
}


def clamp_rect(x0: int, y0: int, size: int, w: int, h: int) -> Tuple[int, int]:
    x0 = max(0, min(w - size, x0))
    y0 = max(0, min(h - size, y0))
    return x0, y0


def crop_centered_square(img: np.ndarray, cx: int, cy: int, size: int) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x0 = int(round(cx - size / 2.0))
    y0 = int(round(cy - size / 2.0))
    x0, y0 = clamp_rect(x0, y0, size, w, h)
    crop = img[y0 : y0 + size, x0 : x0 + size].copy()
    return crop, x0, y0


def extract_pattern_id(filename: str) -> Optional[int]:
    match = re.match(r"^(\d+)_panel_crop\.png$", filename)
    if not match:
        return None
    return int(match.group(1))


def get_gc_status_class(layout: dict) -> Optional[Tuple[str, str]]:
    statuses = layout.get("great_hollow_statuses", [])
    for status in statuses:
        if status.get("anchor_type") == "Great Church":
            status_type = str(status.get("status_type", ""))
            class_name = STATUS_TO_CLASS.get(status_type)
            if class_name is None:
                return None
            return status_type, class_name
    return None


def render_contact_sheet(
    rows: List[dict],
    output_path: Path,
    cols: int = 4,
    title: str = "slot_2 fixed panel-260 bases",
) -> None:
    if not rows:
        return

    ordered = sorted(rows, key=lambda r: (r["class_name"], r["pattern_id"]))
    tiles: List[np.ndarray] = []
    for row in ordered:
        img = cv2.imread(str(row["output_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tile = np.full((320, 300, 3), 245, dtype=np.uint8)
        preview = cv2.resize(img, (260, 260), interpolation=cv2.INTER_AREA)
        tile[50:310, 20:280] = preview
        cv2.putText(
            tile,
            f"pattern {row['pattern_id']}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            row["class_name"],
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return

    n = len(tiles)
    rows_count = int(math.ceil(n / float(cols)))
    sheet = np.full((80 + rows_count * 320, cols * 300, 3), 255, dtype=np.uint8)
    cv2.putText(
        sheet,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        f"count={n} | classes={sorted({r['class_name'] for r in ordered})}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    for idx, tile in enumerate(tiles):
        rr = idx // cols
        cc = idx % cols
        y0 = 80 + rr * 320
        x0 = cc * 300
        sheet[y0 : y0 + 320, x0 : x0 + 300] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fresh slot-2 Great Church status bases (Sleep vs Lightning) from panel crops."
    )
    parser.add_argument(
        "--panel-dir",
        default="tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric/Screenshots/panel_crop",
    )
    parser.add_argument("--layout-json", default="great_hollow_layout_slots_full.json")
    parser.add_argument("--out-dir", default="Testing/slot_2_fixed_geometry_panel_bases260/data/bases260")
    parser.add_argument("--manifest", default="Testing/slot_2_fixed_geometry_panel_bases260/data/bases260_manifest.json")
    parser.add_argument(
        "--contact-sheet",
        default="Testing/slot_2_fixed_geometry_panel_bases260/data/bases260_contact_sheet.png",
    )
    parser.add_argument("--base-size", type=int, default=BASE_SIZE)
    parser.add_argument(
        "--offset-x-px",
        type=int,
        default=SLOT2_OFFSET_X_PX,
        help="Panel-pixel shift applied to slot-2 center before cropping.",
    )
    parser.add_argument(
        "--offset-y-px",
        type=int,
        default=SLOT2_OFFSET_Y_PX,
        help="Panel-pixel shift applied to slot-2 center before cropping.",
    )
    args = parser.parse_args()

    panel_dir = Path(args.panel_dir).resolve()
    layout_json_path = Path(args.layout_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    contact_sheet_path = Path(args.contact_sheet).resolve()
    base_size = int(args.base_size)

    if not panel_dir.exists():
        raise FileNotFoundError(f"Panel directory not found: {panel_dir}")
    if not layout_json_path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")

    payload = json.loads(layout_json_path.read_text(encoding="utf-8"))
    layouts = payload.get("layouts", [])
    layout_by_id: Dict[int, dict] = {int(item["patternId"]): item for item in layouts}

    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in STATUS_TO_CLASS.values():
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        # Override previous bases for this slot-2 dataset.
        for old_png in class_dir.glob("*.png"):
            old_png.unlink()

    rows: List[dict] = []
    skipped: List[dict] = []

    for panel_path in sorted(panel_dir.glob("*_panel_crop.png")):
        pattern_id = extract_pattern_id(panel_path.name)
        if pattern_id is None:
            continue

        layout = layout_by_id.get(pattern_id)
        if layout is None:
            skipped.append(
                {
                    "pattern_id": pattern_id,
                    "panel_path": str(panel_path),
                    "reason": "pattern_id_not_in_layout_json",
                }
            )
            continue

        status_lookup = get_gc_status_class(layout)
        if status_lookup is None:
            skipped.append(
                {
                    "pattern_id": pattern_id,
                    "panel_path": str(panel_path),
                    "reason": "great_church_status_not_sleep_or_lightning",
                }
            )
            continue

        status_type, class_name = status_lookup
        panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        if panel is None:
            skipped.append(
                {
                    "pattern_id": pattern_id,
                    "panel_path": str(panel_path),
                    "reason": "could_not_read_panel_image",
                }
            )
            continue

        h, w = panel.shape[:2]
        cx = int(round(SLOT2_CENTER_NORM_X * w)) + int(args.offset_x_px)
        cy = int(round(SLOT2_CENTER_NORM_Y * h)) + int(args.offset_y_px)
        crop, x0, y0 = crop_centered_square(panel, cx, cy, base_size)

        out_name = f"pattern_{pattern_id}__slot2_panel260_fixed.png"
        out_path = out_dir / class_name / out_name
        cv2.imwrite(str(out_path), crop)

        rows.append(
            {
                "pattern_id": pattern_id,
                "class_name": class_name,
                "status_type": status_type,
                "panel_path": str(panel_path),
                "output_path": str(out_path),
                "panel_shape": {"w": int(w), "h": int(h)},
                "slot2_center_panel_px": {"x": int(cx), "y": int(cy)},
                "crop260_rect": {"x0": int(x0), "y0": int(y0), "size": int(base_size)},
            }
        )

    manifest = {
        "notes": "Fresh slot-2 status bases from panel crops using fixed slot geometry.",
        "slot2_center_norm_top_left": {
            "x_norm": SLOT2_CENTER_NORM_X,
            "y_norm": SLOT2_CENTER_NORM_Y,
        },
        "slot2_center_offset_panel_px": {
            "x": int(args.offset_x_px),
            "y": int(args.offset_y_px),
        },
        "base_size": base_size,
        "status_to_class": STATUS_TO_CLASS,
        "kept_count": len(rows),
        "skipped_count": len(skipped),
        "items": rows,
        "skipped": skipped,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    render_contact_sheet(rows, contact_sheet_path)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    print("Counts by class:", counts)
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
