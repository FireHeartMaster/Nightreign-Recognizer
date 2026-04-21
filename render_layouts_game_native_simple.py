#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

LEFT_BOUND = 507.0
ACTIVE_WIDTH = 1690.0
RAW_MAP_HEIGHT = 1690.0
RAW_GRID_MIN_X = 42
RAW_GRID_MIN_Z = 36
RAW_GRID_SPAN_TILES_X = 4
RAW_GRID_SPAN_TILES_Z = 4
TITLE_FONT_SIZE = 20
TITLE_STROKE = 2

POI_ICON_SIZES = {
    "Mission_Objective.png": {"width": 64},
    "Main_Encampment.png": {"width": 96},
    "Great_Church.png": {"width": 77},
    "Fort.png": {"width": 77},
    "Field_Boss.png": {"width": 64},
    "Field_Boss_Red.png": {"width": 64},
    "Evergaol.png": {"width": 64},
    "Church.png": {"width": 64},
    "Castle.png": {"width": 128},
    "Township.png": {"width": 96},
    "Sorcerer's_Rise.png": {"width": 48},
    "Ruins.png": {"width": 96},
    "Event.png": {"width": 96},
    "Night_Location.png": {"width": 96},
    "Scale_Bearing_Merchant.png": {"width": 48},
    "Spawn_Location.png": {"width": 128},
}

MANUAL_ICON_FALLBACKS = {
    'Map_Event.png': 'Event.png',
    'Arena_Boss.png': 'Field_Boss.png',
    'Sorcerers_Rise.png': "Sorcerer's_Rise.png",
}

@dataclass
class POI:
    x: float
    y: float
    icon: str
    text: str

def csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def as_int(v, default=None):
    if v in ("", None):
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def as_float(v, default=None):
    if v in ("", None):
        return default
    try:
        return float(v)
    except Exception:
        return default

def raw_plot_xy(grid_x: int, grid_z: int, pos_x: float, pos_z: float) -> Tuple[float, float]:
    raw_x = (grid_x - RAW_GRID_MIN_X) * 256.0 + pos_x + 128.0
    raw_z = (grid_z - RAW_GRID_MIN_Z) * 256.0 + pos_z + 128.0
    return raw_x, raw_z

def raw_plot_to_frontend_xy(raw_x: float, raw_z: float) -> Tuple[float, float]:
    x = LEFT_BOUND + (raw_x / (RAW_GRID_SPAN_TILES_X * 256.0)) * ACTIVE_WIDTH
    y = RAW_MAP_HEIGHT - (raw_z / (RAW_GRID_SPAN_TILES_Z * 256.0)) * RAW_MAP_HEIGHT
    return x, y

def frontend_xy_from_raw(grid_x: int, grid_z: int, pos_x: float, pos_z: float) -> Tuple[float, float]:
    raw_x, raw_z = raw_plot_xy(grid_x, grid_z, pos_x, pos_z)
    return raw_plot_to_frontend_xy(raw_x, raw_z)

def map_layout_from_shifting_earth(name: str) -> str:
    return {
        "Mountaintop": "the_mountaintop_shifted",
        "The Crater": "the_crater_shifted",
        "Crater": "the_crater_shifted",
        "Rotted Woods": "the_rotten_woods_shifted",
        "Shrouded City": "noklateo_shifted",
        "Noklateo": "noklateo_shifted",
        "Great Hollow": "great_hollow_shifted",
        "Default": "default",
    }.get(name, "default")

def resolve_icon_path(repo: Path, icon_name: str) -> Optional[Path]:
    poi_dir = repo / "public" / "POI_icons"
    path = poi_dir / icon_name
    if path.exists():
        return path
    fallback = MANUAL_ICON_FALLBACKS.get(icon_name)
    if fallback:
        fpath = poi_dir / fallback
        if fpath.exists():
            return fpath
    return None

def resized_icon(repo: Path, icon_name: str, cache: Dict[str, Image.Image]) -> Optional[Image.Image]:
    if icon_name in cache:
        return cache[icon_name]
    path = resolve_icon_path(repo, icon_name)
    if path is None:
        return None
    img = Image.open(path).convert("RGBA")
    target_width = POI_ICON_SIZES.get(icon_name, {}).get("width", img.width)
    if img.width != target_width:
        scale = target_width / img.width
        img = img.resize((int(round(img.width * scale)), int(round(img.height * scale))), Image.LANCZOS)
    cache[icon_name] = img
    return img

def stitch_map_tiles(repo: Path, map_layout: str) -> Image.Image:
    map_dir = repo / "public" / "assets" / "maps"
    tile_dir = {
        "default": map_dir / "default_map_tiles",
        "noklateo_shifted": map_dir / "noklateo_shifted_map_tiles",
        "the_crater_shifted": map_dir / "the_crater_shifted_map_tiles",
        "the_mountaintop_shifted": map_dir / "the_mountaintop_shifted_map_tiles",
        "the_rotten_woods_shifted": map_dir / "the_rotten_woods_shifted_map_tiles",
        "great_hollow_shifted": map_dir / "great_hollow_shifted_map_tiles",
    }.get(map_layout, map_dir / "default_map_tiles")
    pngs = sorted(tile_dir.glob("*.png"))
    webps = sorted(tile_dir.glob("*.webp"))
    tiles = pngs if pngs else webps
    if not tiles:
        raise FileNotFoundError(f"No tiles found in {tile_dir}")
    # Assume 6x6 like the existing renderer.
    idx_to_img = {}
    for path in tiles:
        stem = path.stem
        try:
            idx = int(stem.split("_")[-1])
        except Exception:
            continue
        idx_to_img[idx] = Image.open(path).convert("RGBA")
    # Fallback if names are just numbers
    if not idx_to_img:
        for path in tiles:
            try:
                idx = int(path.stem)
                idx_to_img[idx] = Image.open(path).convert("RGBA")
            except Exception:
                pass
    if not idx_to_img:
        raise RuntimeError(f"Could not parse tile indexes in {tile_dir}")
    sample = next(iter(idx_to_img.values()))
    w, h = sample.width, sample.height
    canvas = Image.new("RGBA", (w * 6, h * 6), (0, 0, 0, 0))
    for idx, img in idx_to_img.items():
        row = idx // 6
        col = idx % 6
        canvas.alpha_composite(img, dest=(col * w, row * h))
    return canvas.rotate(90, expand=True)

def try_load_font(font_path: Optional[Path], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend([
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ])
    for p in candidates:
        try:
            return ImageFont.truetype(str(p), size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def strip_prefix(value: str) -> str:
    for prefix in [
        "Field Boss - ", "Basement Boss - ", "Evergaol - ", "Castle - ",
        "Fort - ", "Camp - ", "Ruins - ", "Great Church - ", "Church - ",
        "Sorcerer's Rise - ", "Township - ", "Event - ",
    ]:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value

def infer_icon_and_text(key: str, value: str) -> Tuple[Optional[str], str]:
    key_type = key.split(" - ")[0]
    text = strip_prefix(value)

    if key == "Castle - Castle":
        return "Castle.png", text
    if key_type == "Field Boss Red":
        return "Field_Boss_Red.png", text
    if key_type in {"Field Boss", "Arena Boss", "Castle"}:
        return "Field_Boss.png", text
    if key_type == "Evergaol":
        return "Evergaol.png", text
    if key_type == "Map Event":
        return "Event.png", text
    if key_type == "Scale-Bearing Merchant":
        return "Scale_Bearing_Merchant.png", "Scale-Bearing Merchant"
    if key_type == "Major Base":
        if value.startswith("Fort -"):
            return "Fort.png", text
        if value.startswith("Camp -"):
            return "Main_Encampment.png", text
        if value.startswith("Ruins -"):
            return "Ruins.png", text
        if value.startswith("Great Church -"):
            return "Great_Church.png", text
        if value.startswith("Castle -"):
            return "Castle.png", text
        return "Great_Church.png", text
    if key_type == "Minor Base":
        if value.startswith("Church -"):
            return "Church.png", text
        if value.startswith("Sorcerer's Rise"):
            return "Sorcerer's_Rise.png", text
        if value.startswith("Township -"):
            return "Township.png", text
        return "Township.png", text

    return None, text

def extract_pois(layout: dict, attach_by_id: Dict[int, dict], playarea_by_id: Dict[int, dict], sortie_by_modifier: Dict[int, dict]) -> List[POI]:
    pois: List[POI] = []

    # Generic dict entries with attachId/raw ids
    for key, obj in layout.items():
        if not (isinstance(obj, dict) and "location" in obj and "value" in obj):
            continue
        value = obj.get("value")
        if not isinstance(value, str) or value == "empty":
            continue
        icon, text = infer_icon_and_text(key, value)
        if not icon:
            continue
        attach_id = as_int(obj.get("attachId"))
        if attach_id is None:
            gx = as_int(obj.get("gridXNo"))
            gz = as_int(obj.get("gridZNo"))
            px = as_float(obj.get("posX"))
            pz = as_float(obj.get("posZ"))
            if None not in (gx, gz, px, pz):
                x, y = frontend_xy_from_raw(gx, gz, px, pz)
                pois.append(POI(x, y, icon, text))
            continue
        row = attach_by_id.get(attach_id)
        if not row:
            continue
        gx = as_int(row.get("gridXNo"))
        gz = as_int(row.get("gridZNo"))
        px = as_float(row.get("posX"))
        pz = as_float(row.get("posZ"))
        if None in (gx, gz, px, pz):
            continue
        x, y = frontend_xy_from_raw(gx, gz, px, pz)
        pois.append(POI(x, y, icon, text))

    # Spawn point
    spawn = layout.get("Spawn Point")
    if isinstance(spawn, str) and spawn != "empty":
        mod = as_int(layout.get("Spawn Point ID"))
        if mod is None:
            mod = as_int(spawn)
        row = sortie_by_modifier.get(mod) if mod is not None else None
        if row:
            gx = as_int(row.get("spawnGridXNo"))
            gz = as_int(row.get("spawnGridZNo"))
            if None not in (gx, gz):
                x, y = frontend_xy_from_raw(gx, gz, 0.0, 0.0)
                pois.append(POI(x, y, "Spawn_Location.png", "Spawn Point"))

    # Night circles
    for n in (1, 2):
        play_id = as_int(layout.get(f"Night {n} Circle ID"))
        if play_id is None:
            play_id = as_int(layout.get(f"Night {n} Circle"))
        boss_text = str(layout.get(f"Night {n} Boss", f"Night {n} Boss"))
        row = playarea_by_id.get(play_id) if play_id is not None else None
        if row:
            gx = as_int(row.get("gridXNo"))
            gz = as_int(row.get("gridZNo"))
            px = as_float(row.get("posX"))
            pz = as_float(row.get("posZ"))
            if None not in (gx, gz, px, pz):
                x, y = frontend_xy_from_raw(gx, gz, px, pz)
                pois.append(POI(x, y, "Night_Location.png", f"Night {n}\n{boss_text}"))

    return pois

def render_one(repo: Path, layouts_dir: Path, game_csv_dir: Path, layout_number: int, show_text: bool, out_path: Path, font_path: Optional[Path]) -> None:
    layout_path = layouts_dir / f"layout_{layout_number:03d}.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing layout JSON: {layout_path}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    shifting_earth = str(layout.get("Shifting Earth", "Default"))
    map_layout = map_layout_from_shifting_earth(shifting_earth)

    attach_rows = csv_rows(game_csv_dir / "SmallBaseAndSpotAttachPoint.csv")
    playarea_rows = csv_rows(game_csv_dir / "PlayAreaCreateParam.csv")
    sortie_rows = csv_rows(game_csv_dir / "SortieStartingPoint.csv")
    attach_by_id = {as_int(r.get("ID")): r for r in attach_rows if as_int(r.get("ID")) is not None}
    playarea_by_id = {as_int(r.get("ID")): r for r in playarea_rows if as_int(r.get("ID")) is not None}
    sortie_by_modifier = {as_int(r.get("requireModifier")): r for r in sortie_rows if as_int(r.get("requireModifier")) is not None}

    base = stitch_map_tiles(repo, map_layout)
    draw = ImageDraw.Draw(base)
    font = try_load_font(font_path, TITLE_FONT_SIZE)
    cache: Dict[str, Image.Image] = {}

    pois = extract_pois(layout, attach_by_id, playarea_by_id, sortie_by_modifier)
    print(f"[layout {layout_number}] shifting_earth={shifting_earth} map_layout={map_layout} pois={len(pois)}")

    for poi in pois:
        icon = resized_icon(repo, poi.icon, cache)
        if icon is None:
            print(f"Warning: missing icon {poi.icon}")
            continue
        sx, sy = poi.x / 2204.0 * base.width, poi.y / 1690.0 * base.height
        base.alpha_composite(icon, dest=(int(round(sx - icon.width/2)), int(round(sy - icon.height/2))))
        if show_text and poi.text:
            draw.multiline_text(
                (sx, sy + icon.height / 2 + 10),
                poi.text,
                font=font,
                fill=(255,255,255,255),
                stroke_fill=(0,0,0,255),
                stroke_width=TITLE_STROKE,
                anchor="ma",
                align="center",
                spacing=3,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Simple game-native renderer for layout_XXX.json files.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--game-csv-dir", type=Path, required=True)
    ap.add_argument("--layouts-dir", type=Path, default=None)
    ap.add_argument("--layout", type=int, nargs="+", required=True)
    ap.add_argument("--mode", choices=["with_text", "no_text", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--font", type=Path, default=None)
    ap.add_argument("--ext", choices=["png", "jpg", "webp"], default="png")
    args = ap.parse_args()

    layouts_dir = args.layouts_dir or (args.repo / "reference_material" / "pattern_layouts")

    for layout_number in args.layout:
        modes = []
        if args.mode == "both":
            modes = [("no_text", False), ("with_text", True)]
        elif args.mode == "with_text":
            modes = [("with_text", True)]
        else:
            modes = [("no_text", False)]

        for label, show_text in modes:
            out = args.out_dir / f"layout_{layout_number:03d}_{label}.{args.ext}"
            render_one(args.repo, layouts_dir, args.game_csv_dir, layout_number, show_text, out, args.font)

if __name__ == "__main__":
    main()
