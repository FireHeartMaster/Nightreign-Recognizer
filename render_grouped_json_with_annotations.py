#!/usr/bin/env python3
"""
Render grouped LotResultSmallBaseAndSpot JSON directly and save sidecar annotations.

This script is based on the user's current renderer approach:
- input is the grouped JSON (already resolved enough for rendering)
- background is chosen from rareMap
- icon comes from smallBaseLocationName
- text comes from smallBaseVariationName, falling back to smallBaseLocationName

What this version adds:
- renders with_text / no_text / both
- saves one sidecar JSON per rendered image with machine-readable annotations
- saves a combined JSONL manifest for all rendered images
- keeps used_icons.txt / missing_icons.txt

The sidecar JSON is likely the safest extra output for a recognizer pipeline because it records:
- patternId
- rareMap
- image path
- crop info
- every rendered object
- icon name
- text actually drawn
- original world-space coordinates
- image-space coordinates
- icon bounding box
- label bounding box (when text is drawn)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------
# Coordinate system
# Image size: 1536 x 1536
# World bottom-left: (-256, -256)
# World center: (512, 512)
# World top-right: (1280, 1280)
# --------------------------------------------------
IMAGE_W = 1536
IMAGE_H = 1536
WORLD_MIN_X = -256
WORLD_MIN_Z = -256
WORLD_MAX_X = 1280
WORLD_MAX_Z = 1280

ICON_SCALE = 1.8
BASE_ICON_SIZE = 48

SKIP_TEXT_WHEN_ICON_MISSING = False

# Crop only for selected Shifting Earth / rareMap values.
# Key = rareMap name exactly as it appears in the JSON.
# Value = border size in pixels to crop from each side.
CROP_BORDER_BY_RAREMAP = {
    "Great Hollow": 128,
}

FIXED_POINTS_BY_RAREMAP: Dict[str, List[dict]] = {
    "Great Hollow": [
        {"x": 386.488, "z": 926.106, "icon": "Church"},
    ],
    "Noklateo": [
        {"x": 60.5, "z": 60.5, "icon": "Field Boss Red", "text": "Astel"},
        {"x": 204.5, "z": 204.5, "icon": "Field Boss Red", "text": "Dragonkin Soldier"},
        {"x": 254.0, "z": 471.5, "icon": "Field Boss Red", "text": "Royal Carian Knight"},
        {"x": 485.0, "z": 113.0, "icon": "Field Boss Red", "text": "Flying Dragon"},
        {"x": 78.5, "z": 233.0, "icon": "Field Boss", "text": "Black Knife Assassin"},
        {"x": 119.0, "z": 198.5, "icon": "Field Boss", "text": "Royal Revenant"},
        {"x": 228.5, "z": 78.5, "icon": "Field Boss", "text": "Black Knife Assassin"},
        {"x": 227.0, "z": 120.5, "icon": "Field Boss", "text": "Headleass Troll"},
    ],
    "The Crater": [
        {"x": 516.5, "z": 900.5, "icon": "Field Boss", "text": "Fire Prelates"},
        {"x": 360.5, "z": 879.5, "icon": "Field Boss", "text": "Demi-Human Queen"},
        {"x": 291.5, "z": 818.0, "icon": "Field Boss", "text": "Red Wolf"},
        {"x": 669.5, "z": 792.5, "icon": "Field Boss", "text": "Demi-Human Queen"},
        {"x": 353.0, "z": 615.5, "icon": "Field Boss Red", "text": "Valiant Gargoyle"},
        {"x": 522.5, "z": 623.0, "icon": "Field Boss Red", "text": "Fallingstar Beast"},
        {"x": 615.5, "z": 639.5, "icon": "Field Boss Red", "text": "Magma Wyrm"},
        {"x": 365.0, "z": 720.5, "icon": "Church"},
    ],
    "Mountaintop": [
        {"x": 59.0, "z": 570.5, "icon": "Ruins", "text": "Albinauric Archers"},
        {"x": 399.5, "z": 770.0, "icon": "Field Boss", "text": "Snowfield Trolls"},
        {"x": 216.5, "z": 758.0, "icon": "Field Boss", "text": "Flying Dragon"},
        {"x": 122.0, "z": 957.5, "icon": "Field Boss", "text": "Demi-Human Swordmaster"},
        {"x": 426.5, "z": 978.5, "icon": "Field Boss", "text": "Demi-Human Swordmaster"},
        {"x": 255.5, "z": 966.5, "icon": "Field Boss", "text": "Giant Crows"},
        {"x": 104.0, "z": 735.5, "icon": "Field Boss Red", "text": "Mountaintop Ice Dragon"},
        {"x": 35.0, "z": 894.5, "icon": "Field Boss Red", "text": "Mountaintop Ice Dragon"},
        {"x": 296.0, "z": 846.5, "icon": "Field Boss Red", "text": "Mountaintop Ice Dragon"},
        {"x": 338.0, "z": 980.0, "icon": "Field Boss Red", "text": "Mountaintop Ice Dragon"},
    ],
    "Rotted Woods": [
        {"x": 840.5, "z": 144.5, "icon": "Fort", "text": "Lordsworn Captain"},
    ],
}

TEXT_TO_NOT_WRITE = {
    "Church",
}

ROW_RENDER_RULES = [
    {
        "icon_name": "Field Boss",
        "x": 441.0,
        "z": 213.0,
        "render_icon": True,
        "render_text": True,
        "icon_dx_factor": -0.5,
        "icon_dy_factor": 0.5,
        "text_dx_factor": -0.5,
        "text_dy_factor": 0.5,
    },
    {
        "icon_name": "Field Boss Red",
        "x": 441.0,
        "z": 213.0,
        "render_icon": True,
        "render_text": True,
        "icon_dx_factor": -0.5,
        "icon_dy_factor": 0.5,
        "text_dx_factor": -0.5,
        "text_dy_factor": 0.5,
    },
    {
        "icon_name": "Basement Boss",
        "render_icon": True,
        "render_text": True,
        "render_on_top": True,
        "icon_dx_factor": 0.8,
        "icon_dy_factor": 0.0,
        "text_dx_factor": 0.8,
        "text_dy_factor": 0.0,
    },
    {
        "rareMap": "Great Hollow",
        "icon_name": "Great Church",
        "x": 933.46,
        "z": 548.59,
        "render_icon": True,
        "render_text": True,
        "icon_dx_factor": 0.0,
        "icon_dy_factor": -1.5,
        "text_dx_factor": 0.0,
        "text_dy_factor": -1.5,
    },
    {
        "rareMap": "Great Hollow",
        "icon_name": "Great Church",
        "x": 470.74,
        "z": 110.51,
        "render_icon": False,
        "render_text": True,
        "icon_dx_factor": 0.0,
        "icon_dy_factor": 0.0,
        "text_dx_factor": 0.0,
        "text_dy_factor": 0.0,
    },
    {
        "icon_name": "Evergaol",
        "render_layer": 10,
    },
    {
        "icon_name": "Field Boss",
        "render_layer": 20,
    },
    {
        "icon_name": "Field Boss Red",
        "render_layer": 20,
    },
    {
        "icon_name": "Ruins",
        "map_name": "Ruins - Curseblade + Spider Scorpions",
        "x": 138.25,
        "z": 392.67,
        "overlay_icon_name": "Status_Poison",
        "overlay_render_scale": 0.384,
        "overlay_dx_factor": 0.7,
        "overlay_dy_factor": 0.66,
        "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Ruins",
        "map_name": "Ruins - Bloodbane Giant Crows + Chief Bloodfiend",
        "x": 138.25,
        "z": 392.67,
        "overlay_icon_name": "Status_Bleed",
        "overlay_render_scale": 0.384,
        "overlay_dx_factor": 0.7,
        "overlay_dy_factor": 0.66,
        "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Great Church",
        "map_name": "Great Church - Nox Warriors",
        "overlay_icon_name": "Status_Lightning",
        "overlay_render_scale": 0.384,
        "overlay_dx_factor": 0.7,
        "overlay_dy_factor": 0.66,
        "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Great Church",
        "map_name": "Great Church - Abductor Virgins",
        "overlay_icon_name": "Status_Sleep",
        "overlay_render_scale": 0.384,
        "overlay_dx_factor": 0.7,
        "overlay_dy_factor": 0.66,
        "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Night Horde",
        "render_icon": False,
        "render_text": False,
    },
    {
        "map_name": "Sorcerer's Rise - Ruined",
        "icon_name_override": "Sorcerer's Rise Ruined",
    },
    {
        "icon_name": "Small Camp",
        "render_icon": False,
        "render_text": False,
    },
    {
        "rareMap": "Rotted Woods",
        "icon_name": "Church",
        "x": 783.01,
        "z": 350.39,
        "render_icon": False,
        "render_text": False,
    },
]

MISSING_ICON_FILES: set[str] = set()
USED_ICON_FILES: set[str] = set()
ICON_CACHE: Dict[Tuple[str, float], Optional[Image.Image]] = {}


def file_safe_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def float_matches(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def world_to_image(x: float, z: float) -> Tuple[int, int]:
    px = (float(x) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = IMAGE_H - ((float(z) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H)
    px = max(0, min(IMAGE_W - 1, int(round(px))))
    py = max(0, min(IMAGE_H - 1, int(round(py))))
    return px, py


def crop_point(px: float, py: float, crop_px: int) -> Tuple[float, float]:
    return px - crop_px, py - crop_px


def load_font(size: int):
    candidates = [
        "arialbd.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text_by_words(text: str, max_chars: int = 16) -> str:
    text = str(text).strip()
    if text == "":
        return ""
    lines = text.splitlines()
    wrapped_lines = []
    for line in lines:
        line = line.strip()
        if line == "":
            wrapped_lines.append("")
            continue
        wrapped = textwrap.fill(
            line,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False
        )
        wrapped_lines.append(wrapped)
    return "\n".join(wrapped_lines)


def draw_label(draw, center_x, center_y, text, font, fill="white", stroke_fill="black", stroke_width=1, max_chars=16):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)
    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align="center",
        spacing=4,
        stroke_width=stroke_width,
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = center_x - text_w / 2
    y = center_y - text_h / 2
    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        spacing=4,
        align="center",
    )


def get_label_bbox(draw, center_x, center_y, text, font, stroke_width=1, max_chars=16):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)
    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align="center",
        spacing=4,
        stroke_width=stroke_width,
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = center_x - text_w / 2
    y = center_y - text_h / 2
    return wrapped_text, (x, y, x + text_w, y + text_h)


def boxes_overlap(a, b, padding=6):
    return not (
        a[2] + padding <= b[0] or
        b[2] + padding <= a[0] or
        a[3] + padding <= b[1] or
        b[3] + padding <= a[1]
    )


def find_non_overlapping_position(
    draw,
    center_x,
    center_y,
    text,
    font,
    placed_boxes,
    stroke_width=1,
    max_chars=16,
    padding=6,
    step=10,
    max_radius=180,
):
    wrapped_text, bbox = get_label_bbox(
        draw, center_x, center_y, text, font,
        stroke_width=stroke_width,
        max_chars=max_chars
    )
    if not any(boxes_overlap(bbox, old_box, padding=padding) for old_box in placed_boxes):
        return center_x, center_y, bbox

    for radius in range(step, max_radius + step, step):
        candidate_offsets = [
            (0, -radius),
            (radius, 0),
            (0, radius),
            (-radius, 0),
            (radius, -radius),
            (radius, radius),
            (-radius, radius),
            (-radius, -radius),
        ]
        for dx, dy in candidate_offsets:
            test_x = center_x + dx
            test_y = center_y + dy
            _, test_bbox = get_label_bbox(
                draw, test_x, test_y, text, font,
                stroke_width=stroke_width,
                max_chars=max_chars
            )
            if not any(boxes_overlap(test_bbox, old_box, padding=padding) for old_box in placed_boxes):
                return test_x, test_y, test_bbox

    return center_x, center_y, bbox


def paste_centered(base_image, overlay, center_x, center_y):
    x = int(center_x - overlay.width / 2)
    y = int(center_y - overlay.height / 2)
    base_image.paste(overlay, (x, y), overlay)


def normalize_row_icon_name(icon_name: str) -> str:
    icon_name = str(icon_name).strip()
    if icon_name == "":
        return ""
    return icon_name


def get_row_icon(icons_dir: Path, icon_name: str, render_scale: float = 1.0):
    icon_name = normalize_row_icon_name(icon_name)
    if icon_name == "":
        return None

    try:
        render_scale = float(render_scale)
    except (TypeError, ValueError):
        render_scale = 1.0
    if render_scale <= 0:
        render_scale = 1.0

    cache_key = (icon_name, round(render_scale, 4))
    if cache_key in ICON_CACHE:
        return ICON_CACHE[cache_key]

    # Special mapping: use Field_Boss_Red.png for Gold Merchant
    if icon_name == "Gold Merchant":
        icon_path = icons_dir / "Field_Boss_Red.png"
    else:
        icon_path = icons_dir / f"{file_safe_name(icon_name)}.png"

    if not icon_path.exists():
        MISSING_ICON_FILES.add(icon_path.name)
        ICON_CACHE[cache_key] = None
        return None

    USED_ICON_FILES.add(icon_path.name)
    icon_img = Image.open(icon_path).convert("RGBA")

    target_size = int(BASE_ICON_SIZE * ICON_SCALE * render_scale)
    orig_w, orig_h = icon_img.size
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    ICON_CACHE[cache_key] = icon_img
    return icon_img


def get_row_render_config(rare_map, icon_name, variation_name, map_name, x_world, z_world):
    config = {
        "render_icon": True,
        "render_text": icon_name not in TEXT_TO_NOT_WRITE,
        "render_on_top": False,
        "render_layer": 0,
        "render_scale": 1.0,
        "icon_dx": 0,
        "icon_dy": 0,
        "text_dx": 0,
        "text_dy": 0,
        "icon_dx_factor": 0.0,
        "icon_dy_factor": 0.0,
        "text_dx_factor": 0.0,
        "text_dy_factor": 0.0,
        "overlay_icon_name": "",
        "overlay_render_scale": 1.0,
        "overlay_dx": 0,
        "overlay_dy": 0,
        "overlay_dx_factor": 0.0,
        "overlay_dy_factor": 0.0,
        "overlay_layer_offset": 1,
        "overlay_render_on_top": False,
        "icon_name_override": "",
    }

    for rule in ROW_RENDER_RULES:
        rule_rare_map = str(rule.get("rareMap", "")).strip()
        if rule_rare_map != "" and str(rare_map).strip() != rule_rare_map:
            continue

        rule_icon_name = str(rule.get("icon_name", "")).strip()
        if rule_icon_name != "" and icon_name != rule_icon_name:
            continue

        rule_variation_name = str(rule.get("variation_name", "")).strip()
        if rule_variation_name != "" and variation_name != rule_variation_name:
            continue

        rule_map_name = str(rule.get("map_name", "")).strip()
        if rule_map_name != "" and map_name != rule_map_name:
            continue

        rule_map_name_in = rule.get("map_name_in", None)
        if rule_map_name_in is not None:
            normalized_list = {str(v).strip() for v in rule_map_name_in}
            if map_name not in normalized_list:
                continue

        if "x" in rule and not float_matches(x_world, rule["x"]):
            continue
        if "z" in rule and not float_matches(z_world, rule["z"]):
            continue

        for key in config.keys():
            if key in rule:
                config[key] = rule[key]

    return config


def annotation_from_icon(
    source_type: str,
    rare_map: str,
    world_x: float,
    world_z: float,
    center_px: float,
    center_py: float,
    icon_name: str,
    icon_img: Image.Image,
    text: str,
    render_layer: int,
    crop_px: int,
    extra: Optional[dict] = None,
):
    left = center_px - icon_img.width / 2
    top = center_py - icon_img.height / 2
    bbox = [left, top, left + icon_img.width, top + icon_img.height]
    if crop_px > 0:
        bbox = [bbox[0] - crop_px, bbox[1] - crop_px, bbox[2] - crop_px, bbox[3] - crop_px]
        center_out = [center_px - crop_px, center_py - crop_px]
    else:
        center_out = [center_px, center_py]

    payload = {
        "sourceType": source_type,
        "rareMap": rare_map,
        "world": {"x": float(world_x), "z": float(world_z)},
        "imageCenter": {"x": float(center_out[0]), "y": float(center_out[1])},
        "iconName": icon_name,
        "text": text,
        "renderLayer": int(render_layer),
        "iconBBox": [float(v) for v in bbox],
    }
    if extra:
        payload.update(extra)
    return payload


def apply_crop_to_bbox(bbox, crop_px):
    if bbox is None:
        return None
    if crop_px <= 0:
        return [float(v) for v in bbox]
    return [float(bbox[0] - crop_px), float(bbox[1] - crop_px), float(bbox[2] - crop_px), float(bbox[3] - crop_px)]


def render_pattern(
    entry: dict,
    backgrounds_dir: Path,
    icons_dir: Path,
    output_dir: Path,
    mode: str,
    title_font,
    boss_font,
    manifest_records: list,
):
    pattern_id = entry.get("patternId", "")
    rare_map = str(entry.get("rareMap", "")).strip()
    if rare_map == "":
        return

    background_path = backgrounds_dir / f"{file_safe_name(rare_map)}.png"
    if not background_path.exists():
        return

    bg_base = Image.open(background_path).convert("RGBA")
    if bg_base.size != (IMAGE_W, IMAGE_H):
        bg_base = bg_base.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)

    boss_icon_path = icons_dir / "Night_location.png"
    if not boss_icon_path.exists():
        raise FileNotFoundError(f"Missing boss icon: {boss_icon_path}")
    USED_ICON_FILES.add(boss_icon_path.name)
    boss_icon = Image.open(boss_icon_path).convert("RGBA")
    target_size = int(BASE_ICON_SIZE * ICON_SCALE)
    orig_w, orig_h = boss_icon.size
    scale = target_size / max(orig_w, orig_h)
    boss_icon = boss_icon.resize(
        (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))),
        Image.Resampling.LANCZOS
    )

    modes = ["no_text", "with_text"] if mode == "both" else [mode]

    for one_mode in modes:
        render_any_text = (one_mode == "with_text")
        bg = bg_base.copy()
        draw = ImageDraw.Draw(bg)

        icon_queue = []
        label_queue = []
        annotations = []

        crop_px = int(CROP_BORDER_BY_RAREMAP.get(rare_map, 0))

        # fixed points
        for point in FIXED_POINTS_BY_RAREMAP.get(rare_map, []):
            icon_name = str(point.get("icon", "")).strip()
            text_value = str(point.get("text", "") or "").strip()
            x_world = point.get("x", None)
            z_world = point.get("z", None)
            if icon_name == "" or x_world is None or z_world is None:
                continue

            config = get_row_render_config(
                rare_map=rare_map,
                icon_name=icon_name,
                variation_name="",
                map_name="",
                x_world=x_world,
                z_world=z_world,
            )

            icon_name_effective = str(config["icon_name_override"]).strip() or icon_name
            icon_img = None
            if config["render_icon"]:
                icon_img = get_row_icon(icons_dir, icon_name_effective, config["render_scale"])
                if icon_img is None and SKIP_TEXT_WHEN_ICON_MISSING:
                    continue

            px, py = world_to_image(x_world, z_world)
            icon_px = px + config["icon_dx"]
            icon_py = py + config["icon_dy"]
            text_px = px + config["text_dx"]
            text_py = py + config["text_dy"]

            render_layer = int(config["render_layer"])
            if config["render_on_top"]:
                render_layer = max(render_layer, 1000)

            if icon_img is not None:
                icon_px += int(round(icon_img.width * config["icon_dx_factor"]))
                icon_py += int(round(icon_img.height * config["icon_dy_factor"]))
                text_px += int(round(icon_img.width * config["text_dx_factor"]))
                text_py += int(round(icon_img.height * config["text_dy_factor"]))

                icon_queue.append((render_layer, icon_img, icon_px, icon_py))
                annotations.append(annotation_from_icon(
                    "fixed_point", rare_map, x_world, z_world, icon_px, icon_py,
                    icon_name_effective, icon_img, text_value,
                    render_layer, crop_px,
                    {"fixedPoint": point}
                ))

            if render_any_text and config["render_text"] and text_value != "":
                label_queue.append({
                    "x": text_px,
                    "y": text_py,
                    "text": text_value,
                    "font": boss_font,
                    "sourceType": "fixed_point",
                    "rareMap": rare_map,
                    "world": {"x": float(x_world), "z": float(z_world)},
                    "iconName": icon_name_effective,
                })

        # row items
        for row in entry.get("rows", []):
            icon_name = str(row.get("smallBaseLocationName", "") or "").strip()
            variation_name = str(row.get("smallBaseVariationName", "") or "").strip()
            map_name = str(row.get("smallBaseMapName", "") or "").strip()
            x_world = row.get("X", None)
            z_world = row.get("Z", None)

            if icon_name == "" or x_world in ("", None) or z_world in ("", None):
                continue

            config = get_row_render_config(
                rare_map=rare_map,
                icon_name=icon_name,
                variation_name=variation_name,
                map_name=map_name,
                x_world=x_world,
                z_world=z_world,
            )

            if not config["render_icon"] and not config["render_text"] and str(config["overlay_icon_name"]).strip() == "":
                continue

            icon_name_effective = str(config["icon_name_override"]).strip() or icon_name
            text_to_draw = variation_name if variation_name != "" else icon_name

            icon_img = None
            if config["render_icon"]:
                icon_img = get_row_icon(icons_dir, icon_name_effective, config["render_scale"])
                if icon_img is None and SKIP_TEXT_WHEN_ICON_MISSING:
                    continue

            px, py = world_to_image(x_world, z_world)
            icon_px = px + config["icon_dx"]
            icon_py = py + config["icon_dy"]
            text_px = px + config["text_dx"]
            text_py = py + config["text_dy"]

            base_render_layer = int(config["render_layer"])
            if config["render_on_top"]:
                base_render_layer = max(base_render_layer, 1000)

            if icon_img is not None:
                icon_px += int(round(icon_img.width * config["icon_dx_factor"]))
                icon_py += int(round(icon_img.height * config["icon_dy_factor"]))
                text_px += int(round(icon_img.width * config["text_dx_factor"]))
                text_py += int(round(icon_img.height * config["text_dy_factor"]))

                icon_queue.append((base_render_layer, icon_img, icon_px, icon_py))
                annotations.append(annotation_from_icon(
                    "row", rare_map, x_world, z_world, icon_px, icon_py,
                    icon_name_effective, icon_img, text_to_draw,
                    base_render_layer, crop_px,
                    {"rowData": row}
                ))

            overlay_icon_name = str(config["overlay_icon_name"]).strip()
            if overlay_icon_name != "":
                overlay_img = get_row_icon(icons_dir, overlay_icon_name, config["overlay_render_scale"])
                if overlay_img is not None:
                    overlay_px = icon_px + config["overlay_dx"]
                    overlay_py = icon_py + config["overlay_dy"]
                    overlay_px += int(round(overlay_img.width * config["overlay_dx_factor"]))
                    overlay_py += int(round(overlay_img.height * config["overlay_dy_factor"]))
                    overlay_layer = base_render_layer + int(config["overlay_layer_offset"])
                    if config["overlay_render_on_top"]:
                        overlay_layer = max(overlay_layer, 1001)

                    icon_queue.append((overlay_layer, overlay_img, overlay_px, overlay_py))
                    annotations.append(annotation_from_icon(
                        "row_overlay", rare_map, x_world, z_world, overlay_px, overlay_py,
                        overlay_icon_name, overlay_img, "",
                        overlay_layer, crop_px,
                        {"parentRowData": row}
                    ))

            if render_any_text and config["render_text"] and text_to_draw != "":
                label_queue.append({
                    "x": text_px,
                    "y": text_py,
                    "text": text_to_draw,
                    "font": boss_font,
                    "sourceType": "row",
                    "rareMap": rare_map,
                    "world": {"x": float(x_world), "z": float(z_world)},
                    "iconName": icon_name_effective,
                    "rowData": row,
                })

        # night bosses at play areas
        for night_idx, boss_name_key, x_key, z_key in [
            (1, "bossId1Name", "playArea1_X", "playArea1_Z"),
            (2, "bossId2Name", "playArea2_X", "playArea2_Z"),
        ]:
            x_world = entry.get(x_key, None)
            z_world = entry.get(z_key, None)
            if x_world in ("", None) or z_world in ("", None):
                continue

            boss_name = str(entry.get(boss_name_key, "") or "").strip()
            if boss_name == "":
                continue

            extra_key = "extraBossId1Name" if night_idx == 1 else "extraBossId2Name"
            extra_name = str(entry.get(extra_key, "") or "").strip()
            if extra_name != "":
                text_value = f"Night {night_idx}\n{boss_name}\nExtra: {extra_name}"
            else:
                text_value = f"Night {night_idx}\n{boss_name}"

            px, py = world_to_image(x_world, z_world)
            icon_queue.append((200, boss_icon, px, py))
            annotations.append(annotation_from_icon(
                "night_boss", rare_map, x_world, z_world, px, py,
                "Night_location", boss_icon, text_value, 200, crop_px,
                {"nightIndex": night_idx}
            ))

            if render_any_text:
                label_queue.append({
                    "x": px,
                    "y": py,
                    "text": text_value,
                    "font": boss_font,
                    "sourceType": "night_boss",
                    "rareMap": rare_map,
                    "world": {"x": float(x_world), "z": float(z_world)},
                    "iconName": "Night_location",
                    "nightIndex": night_idx,
                })

        # draw icons
        for _, icon_img, px, py in sorted(icon_queue, key=lambda item: item[0]):
            paste_centered(bg, icon_img, px, py)

        # draw text and record final label bboxes
        if render_any_text:
            placed_label_boxes = []
            seen_labels = set()
            label_annotations = []

            for item in label_queue:
                x = item["x"]
                y = item["y"]
                text = item["text"]
                font = item["font"]

                label_key = (round(float(x), 3), round(float(y), 3), str(text).strip(), id(font))
                if label_key in seen_labels:
                    continue
                seen_labels.add(label_key)

                final_x, final_y, final_bbox = find_non_overlapping_position(
                    draw=draw,
                    center_x=x,
                    center_y=y,
                    text=text,
                    font=font,
                    placed_boxes=placed_label_boxes,
                    stroke_width=1,
                    max_chars=16,
                    padding=6,
                    step=10,
                    max_radius=180,
                )

                draw_label(draw, final_x, final_y, text, font)
                placed_label_boxes.append(final_bbox)

                label_bbox_out = apply_crop_to_bbox(final_bbox, crop_px)
                final_center = [float(final_x - crop_px if crop_px > 0 else final_x),
                                float(final_y - crop_px if crop_px > 0 else final_y)]

                label_annotations.append({
                    "sourceType": item["sourceType"],
                    "rareMap": item["rareMap"],
                    "world": item["world"],
                    "iconName": item["iconName"],
                    "text": text,
                    "labelCenter": {"x": final_center[0], "y": final_center[1]},
                    "labelBBox": label_bbox_out,
                    **({k: v for k, v in item.items() if k not in {"x", "y", "text", "font", "sourceType", "rareMap", "world", "iconName"}})
                })
        else:
            label_annotations = []

        if crop_px > 0:
            if crop_px * 2 >= bg.width or crop_px * 2 >= bg.height:
                raise ValueError(
                    f"Crop for rareMap '{rare_map}' is too large: {crop_px}px for image size {bg.width}x{bg.height}"
                )
            bg = bg.crop((crop_px, crop_px, bg.width - crop_px, bg.height - crop_px))

        out_path = output_dir / one_mode / f"pattern_{int(pattern_id):04d}_{file_safe_name(rare_map)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(out_path)

        sidecar = {
            "patternId": int(pattern_id),
            "rareMap": rare_map,
            "mode": one_mode,
            "outputImage": str(out_path),
            "backgroundImage": str(background_path),
            "cropBorder": crop_px,
            "outputImageSize": {"width": bg.width, "height": bg.height},
            "bossId1Name": entry.get("bossId1Name", ""),
            "bossId2Name": entry.get("bossId2Name", ""),
            "extraBossId1Name": entry.get("extraBossId1Name", ""),
            "extraBossId2Name": entry.get("extraBossId2Name", ""),
            "playAreas": {
                "playArea1": {"x": entry.get("playArea1_X", ""), "z": entry.get("playArea1_Z", "")},
                "playArea2": {"x": entry.get("playArea2_X", ""), "z": entry.get("playArea2_Z", "")},
            },
            "iconAnnotations": annotations,
            "labelAnnotations": label_annotations,
        }

        sidecar_path = out_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest_records.append(sidecar)


def main():
    parser = argparse.ArgumentParser(description="Render grouped JSON directly and save sidecar annotations.")
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--backgrounds-dir", type=Path, required=True)
    parser.add_argument("--icons-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["with_text", "no_text", "both"], default="both")
    parser.add_argument("--pattern-ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    title_font = load_font(28)
    boss_font = load_font(22)

    wanted = set(args.pattern_ids) if args.pattern_ids else None
    manifest_records = []

    for entry in data:
        pattern_id = entry.get("patternId", None)
        if pattern_id in ("", None):
            continue
        try:
            pattern_id_int = int(pattern_id)
        except Exception:
            continue
        if wanted is not None and pattern_id_int not in wanted:
            continue

        render_pattern(
            entry=entry,
            backgrounds_dir=args.backgrounds_dir,
            icons_dir=args.icons_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            title_font=title_font,
            boss_font=boss_font,
            manifest_records=manifest_records,
        )

    missing_icons_txt = args.output_dir / "missing_icons.txt"
    used_icons_txt = args.output_dir / "used_icons.txt"
    manifest_jsonl = args.output_dir / "render_manifest.jsonl"

    with missing_icons_txt.open("w", encoding="utf-8") as f:
        for name in sorted(MISSING_ICON_FILES):
            f.write(f"{name}\n")

    with used_icons_txt.open("w", encoding="utf-8") as f:
        for name in sorted(USED_ICON_FILES):
            f.write(f"{name}\n")

    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Rendered {len(manifest_records)} image(s)")
    print(f"Saved manifest: {manifest_jsonl}")
    print(f"Saved missing icon list: {missing_icons_txt}")
    print(f"Saved used icon list: {used_icons_txt}")


if __name__ == "__main__":
    main()
