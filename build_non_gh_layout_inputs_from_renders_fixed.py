#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


NON_GH_EARTHS = ("Limveld", "Noklateo", "The Crater", "Mountaintop", "Rotted Woods")
PANEL_SIZE = 1536
FURTHER_CROP_BORDER = 256
FURTHER_SIZE = PANEL_SIZE - (2 * FURTHER_CROP_BORDER)

# Slot-capable landmark types for non-Great-Hollow layouts.
SLOT_TYPES = {
    "Fort",
    "Fortified Village",
    "Great Church",
    "Main Encampment",
    "Ruins",
    "Swamp",
    "Church",
    "Sorcerer's Rise",
    "Township",
}


def earth_slug(earth: str) -> str:
    return earth.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def normalize_location_name(name: str) -> str:
    value = str(name or "").strip()
    if value == "Sorcerer's Rise Ruined":
        return "Sorcerer's Rise"
    if value == "Camp":
        return "Main Encampment"
    return value


def ensure_panel_size(img):
    h, w = img.shape[:2]
    if (w, h) == (PANEL_SIZE, PANEL_SIZE):
        return img
    return cv2.resize(img, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)


def further_crop(panel_img):
    b = int(FURTHER_CROP_BORDER)
    return panel_img[b : PANEL_SIZE - b, b : PANEL_SIZE - b]


def xy_key(x: float, z: float) -> Tuple[float, float]:
    return (round(float(x), 3), round(float(z), 3))


def make_slot_row(slot_id: int, x_bl: float, y_bl: float, label: str, allowed_types: List[str]) -> dict:
    return {
        "slot_id": int(slot_id),
        "radius_norm": 0.018,
        "allowed_types": list(allowed_types),
        "label": str(label),
        "x_further_px_bl": float(x_bl),
        "y_further_px_bl": float(y_bl),
        "x_further_norm_bl": float(x_bl) / float(FURTHER_SIZE),
        "y_further_norm_bl": float(y_bl) / float(FURTHER_SIZE),
        "inside_further_crop": bool(0.0 <= x_bl <= FURTHER_SIZE and 0.0 <= y_bl <= FURTHER_SIZE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build non-GH panel/further-crop training inputs using ground truth from "
            "LotResultSmallBaseAndSpot_grouped_by_patternId_16.json."
        )
    )
    parser.add_argument(
        "--source-images",
        default="renders_fixed/recognizer_no_text",
        help="Directory with layout_XXXX.png images.",
    )
    parser.add_argument(
        "--ground-truth-json",
        default="LotResultSmallBaseAndSpot_grouped_by_patternId_16.json",
        help="Ground-truth JSON with rows (X/Z + smallBaseLocationName).",
    )
    parser.add_argument(
        "--output-root",
        default="tmp_v16_non_gh_layout_inputs_from_renders_fixed",
        help="Output root for generated crops and strict metadata.",
    )
    args = parser.parse_args()

    source_images = Path(args.source_images)
    gt_json = Path(args.ground_truth_json)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with gt_json.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    entries_by_earth: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        earth = str(e.get("rareMap", "")).strip()
        if earth in NON_GH_EARTHS:
            entries_by_earth[earth].append(e)

    summary = {
        "source_images": str(source_images.resolve()),
        "ground_truth_json": str(gt_json.resolve()),
        "output_root": str(output_root.resolve()),
        "earths": {},
    }

    for earth in NON_GH_EARTHS:
        earth_entries = sorted(entries_by_earth.get(earth, []), key=lambda x: int(x.get("patternId", -1)))
        slug = earth_slug(earth)
        earth_root = output_root / slug
        panel_dir = earth_root / "panel_crop_1536"
        further_dir = earth_root / "further_crop_1024"
        panel_dir.mkdir(parents=True, exist_ok=True)
        further_dir.mkdir(parents=True, exist_ok=True)

        allowed_types_seen: set = set()
        slot_points: set = set()
        for e in earth_entries:
            for row in e.get("rows", []):
                t = normalize_location_name(row.get("smallBaseLocationName", ""))
                if t not in SLOT_TYPES:
                    continue
                x = float(row.get("X", 0.0))
                z = float(row.get("Z", 0.0))
                slot_points.add(xy_key(x, z))
                allowed_types_seen.add(t)

        sorted_slot_points = sorted(slot_points, key=lambda p: (-p[1], p[0]))
        slot_id_by_key = {k: i + 1 for i, k in enumerate(sorted_slot_points)}
        allowed_types_sorted = sorted(allowed_types_seen)
        slot_defs = [
            make_slot_row(
                slot_id=slot_id_by_key[k],
                x_bl=k[0],
                y_bl=k[1],
                label="empty",
                allowed_types=allowed_types_sorted,
            )
            for k in sorted_slot_points
        ]

        missing_images: List[int] = []
        duplicate_slot_hits: Dict[int, List[dict]] = defaultdict(list)
        layouts_payload: List[dict] = []
        written_pids: List[int] = []

        for e in earth_entries:
            pid = int(e.get("patternId", -1))
            src_img = source_images / f"layout_{pid:04d}.png"
            if not src_img.exists():
                missing_images.append(pid)
                continue

            img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
            if img is None:
                missing_images.append(pid)
                continue

            panel_img = ensure_panel_size(img)
            crop_img = further_crop(panel_img)
            panel_out = panel_dir / f"pattern_{pid:04d}_{slug}_panel_crop_1536.png"
            further_out = further_dir / f"pattern_{pid:04d}_{slug}_further_crop_1024.png"
            cv2.imwrite(str(panel_out), panel_img)
            cv2.imwrite(str(further_out), crop_img)
            written_pids.append(pid)

            label_by_slot_key: Dict[Tuple[float, float], str] = {}
            for row in e.get("rows", []):
                t = normalize_location_name(row.get("smallBaseLocationName", ""))
                if t not in SLOT_TYPES:
                    continue
                key = xy_key(float(row.get("X", 0.0)), float(row.get("Z", 0.0)))
                if key not in slot_id_by_key:
                    continue
                if key in label_by_slot_key and label_by_slot_key[key] != t:
                    duplicate_slot_hits[pid].append(
                        {
                            "slot_key": [key[0], key[1]],
                            "existing": label_by_slot_key[key],
                            "incoming": t,
                        }
                    )
                    continue
                label_by_slot_key[key] = t

            slots_payload = []
            for key in sorted_slot_points:
                slot_id = slot_id_by_key[key]
                label = label_by_slot_key.get(key, "empty")
                slots_payload.append(
                    make_slot_row(
                        slot_id=slot_id,
                        x_bl=key[0],
                        y_bl=key[1],
                        label=label,
                        allowed_types=allowed_types_sorted,
                    )
                )

            layouts_payload.append(
                {
                    "patternId": pid,
                    "slots": slots_payload,
                    # Keep landmark list non-empty for libs that stack landmark descriptors.
                    "landmarks": [dict(s) for s in slots_payload],
                }
            )

        strict_json = output_root / f"{slug}_layout_slots_further_crop_bl_strict_v16.json"
        payload = {
            "source_file": str(gt_json.resolve()),
            "coordinate_system": {
                "target": {
                    "name": "further_crop",
                    "size": {"width": FURTHER_SIZE, "height": FURTHER_SIZE},
                    "origin": "bottom_left",
                    "fields": [
                        "x_further_px_bl",
                        "y_further_px_bl",
                        "x_further_norm_bl",
                        "y_further_norm_bl",
                    ],
                    "crop_from_panel_px": {
                        "left": FURTHER_CROP_BORDER,
                        "right": FURTHER_CROP_BORDER,
                        "top": FURTHER_CROP_BORDER,
                        "bottom": FURTHER_CROP_BORDER,
                    },
                },
                "note": (
                    "Ground truth from LotResult..._16 rows. "
                    "Coordinates are expressed as further-crop bottom-left."
                ),
            },
            "shifting_earth": earth,
            "slot_count": len(sorted_slot_points),
            "layouts_count": len(layouts_payload),
            "slot_definitions": slot_defs,
            "layouts": layouts_payload,
        }
        with strict_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        summary["earths"][earth] = {
            "pattern_count_expected": len(earth_entries),
            "pattern_count_written": len(written_pids),
            "slot_count": len(sorted_slot_points),
            "missing_images": missing_images,
            "duplicate_slot_hits": duplicate_slot_hits,
            "panel_dir": str(panel_dir.resolve()),
            "further_dir": str(further_dir.resolve()),
            "strict_metadata_json": str(strict_json.resolve()),
        }

        print(
            f"[{earth}] patterns={len(written_pids)}/{len(earth_entries)} "
            f"slots={len(sorted_slot_points)} metadata={strict_json.name}"
        )

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
