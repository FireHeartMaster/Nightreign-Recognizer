#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Raw-game -> router-like plotting constants
LEFT_BOUND = 507.0
ACTIVE_WIDTH = 1690.0
RAW_MAP_HEIGHT = 1690.0
RAW_GRID_MIN_X = 42
RAW_GRID_MIN_Z = 36
RAW_GRID_SPAN_TILES_X = 4
RAW_GRID_SPAN_TILES_Z = 4

TITLE_FONT_SIZE = 20
TITLE_STROKE = 2

SHIFTING_EARTH_BY_MODIFIER = {
    11: "Mountaintop",
    12: "The Crater",
    13: "Rotted Woods",
    14: "Great Hollow",
    15: "Shrouded City",
}

# Known/confirmed smallBaseMapId -> text mapping from this chat.
KNOWN_SMALLBASE_NAMES: Dict[int, str] = {
    3000: "Fort - Lordsworn Captain; Abductor Virgin",
    3030: "Fort - Guardian Golem; Crystalians",
    3200: "Camp - Banished Knights; Elder Lion",
    3210: "Camp - Leonine Misbegotten; Redmane Knights; Royal Army Knights",
    3220: "Camp - Flame Chariots; Frenzied Flame Troll",
    3400: "Ruins - Erdtree Burial Watchdogs; Sanguine Noble; Ancient Heroes of Zamor; Large Albinaurics",
    3410: "Ruins - Perfumer; Depraved Perfumer; Battlemages; Albinauric Archers; Runebear",
    3420: "Ruins - Wormface; Beastmen of Farum Azula",
    3430: "Ruins - Beastmen of Farum Azula",
    3800: "Great Church - Oracle Envoys; Guardian Golem",
    3810: "Great Church - Mausoleum Knight; Fire Monk",
    4000: "Sorcerer's Rise - Normal",
    4090: "Sorcerer's Rise - Ruined",
    4100: "Church - Dogs",
    4101: "Church - Rats; Rats Alt",
    4551: "Field Boss - Black Knife Assassin",
    4552: "Event - Maris",
    4553: "Event - Gnoster",
    4555: "Event - Libra",
    4650: "Evergaol - Unused Banished Knights; Unused Nox Duo; Omen; Grave Warden Duelist; Beastmen of Farum Azula; Beastly Brigade; Unused Bloodhound Knight",
    4651: "Field Boss - Red Wolf",
    4652: "Field Boss - Draconic Tree Sentinel",
    4653: "Field Boss - Tree Sentinel",
    4654: "Field Boss - Royal Carian Knight",
    4655: "Field Boss - Leonine Misbegotten",
    4656: "Field Boss - Bell Bearing Hunter",
    4657: "Field Boss - Elder Lion",
    4658: "Field Boss - Flying Dragon of the Hills",
    4659: "Field Boss - Royal Revenant",
    4660: "Evergaol - Crucible Knight with Sword; Crucible Knight with Spear; Crystalians; Stoneskin Lords; Banished Knights; Nox Duo; Bloodhound Knight",
    4662: "Field Boss - Night's Cavalry",
    4663: "Field Boss - Ulcerated Tree Spirit",
    4664: "Field Boss - Ancestor Spirit",
    4665: "Field Boss - Grafted Scion",
    4666: "Field Boss - Valiant Gargoyle",
    4667: "Field Boss - Erdtree Avatar",
    4668: "Field Boss - Magma Wyrm",
    4669: "Field Boss - Ancient Hero of Zamor",
    4670: "Evergaol - Dragonkin Soldier; Godskin Apostle; Godskin Noble; Crucible Knight with Sword; Crucible Knight with Spear",
    4671: "Field Boss - Miranda Blossom",
    4672: "Field Boss - Golden Hippopotamus",
    4674: "Field Boss - Death Rite Bird",
    4677: "Field Boss - Demi-Human Queen",
    4678: "Event - Fell Omen",
    4681: "Basement Boss - Black Knife Assassin",
    4682: "Basement Boss - Red Wolf",
    4686: "Basement Boss - Leonine Misbegotten",
    4687: "Basement Boss - Bell Bearing Hunter",
    4688: "Basement Boss - Royal Revenant",
    4691: "Basement Boss - Grafted Scion",
    4695: "Basement Boss - Ancient Hero of Zamor",
    4770: "Tibia Mariner / Those Who Live in Death",
    4780: "Gaping Dragon",
    4790: "Centipede Demon",
    4800: "The Duke's Dear Freja",
    4810: "Smelter Demon",
    4820: "Nameless King",
    4830: "Dancer of the Boreal Valley",
    4840: "Fell Omen (Morgott)",
    4850: "Draconic Tree Sentinel / Royal Cavalrymen",
    4860: "Tree Sentinel / Royal Cavalrymen",
    4880: "Godskin Duo",
    4890: "Wormface",
    4900: "Tibia Mariner / Those Who Live in Death",
    4917: "Valiant Gargoyle / Black Blade Kindred",
    4918: "Magma Wyrm / Great Wyrm",
    4919: "Ancient Dragon",
    4920: "Full-Grown Fallingstar Beast",
    4921: "Death Rite Bird",
    4923: "Nox Dragonkin Soldier",
    4924: "Bell Bearing Hunter family",
    4925: "Crucible Knight / Golden Hippopotamus family",
    4926: "Outland Commander (Niall)",
    4927: "Battlefield Commander (O'Neil)",
    4928: "Night's Cavalry (Duo)",
    4929: "Demi-Human Queen / Demi-Human Swordmaster family",
    4930: "Royal Revenant family",
    4940: "Event - Libra Alt",
    4941: "Castle - Banished Knights",
    4942: "Castle - Trolls",
    4943: "Castle - Crucible Knights",
    4990: "Ulcerated Tree Spirit",
    5100: "Church - Normal",
    5110: "Sorcerer's Rise",
    # Great Hollow/DLC-side known confirmations can be added over time.
}

BOSS_ID_TO_NAME = {
    400: "Fell Omen (Morgott)",
    402: "Tree Sentinel / Royal Cavalrymen",
    403: "Draconic Tree Sentinel / Royal Cavalrymen",
    406: "Royal Revenant family",
    412: "Wormface",
    413: "Ulcerated Tree Spirit",
    416: "Valiant Gargoyle / Black Blade Kindred",
    418: "Magma Wyrm / Great Wyrm",
    427: "Gaping Dragon",
    428: "Centipede Demon",
    429: "The Duke's Dear Freja",
    430: "Smelter Demon",
    431: "Nameless King",
    432: "Dancer of the Boreal Valley",
    434: "Godskin Duo",
    435: "Full-Grown Fallingstar Beast",
    437: "Tibia Mariner / Those Who Live in Death",
    439: "Bell Bearing Hunter family",
    440: "Night's Cavalry (Duo)",
    443: "Death Rite Bird",
    444: "Demi-Human Queen / Demi-Human Swordmaster family",
    445: "Ancient Dragon",
    446: "Nox Dragonkin Soldier",
    447: "Crucible Knight / Golden Hippopotamus family",
    448: "Battlefield Commander (O'Neil)",
    449: "Outland Commander (Niall)",
    5200: "Great Hollow Day 1 Boss 5200",
    5201: "Great Hollow Day 1 Boss 5201",
    5202: "Great Hollow Day 1 Boss 5202",
    5203: "Great Hollow Day 1 Boss 5203",
    5210: "Great Hollow Day 2 Boss 5210",
    5211: "Great Hollow Day 2 Boss 5211",
    5212: "Great Hollow Day 2 Boss 5212",
    5213: "Great Hollow Day 2 Boss 5213",
}

FIELD_BOSS_RED_IDS = {
    4652, 4653, 4654, 4656, 4658, 4663, 4664, 4666, 4667, 4668, 4674,
}
FIELD_BOSS_IDS = {
    4551, 4651, 4655, 4657, 4659, 4662, 4665, 4669, 4671, 4672, 4677,
}
BASEMENT_BOSS_IDS = {4681, 4682, 4686, 4687, 4688, 4691, 4695}
EVENT_IDS = {4552, 4553, 4555, 4678, 4940}
MAJOR_PREFIXES = ("Fort -", "Camp -", "Ruins -", "Great Church -", "Castle -")
MINOR_PREFIXES = ("Church -", "Sorcerer's Rise", "Township -")

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
    "Map_Event.png": "Event.png",
    "Arena_Boss.png": "Field_Boss.png",
    "Sorcerers_Rise.png": "Sorcerer's_Rise.png",
}

TILE_PATTERNS = [
    re.compile(r"MENU_MapTile_L(?P<level>\d+)_(?P<row>\d+)_(?P<col>\d+)_webp\.(png|webp)$", re.IGNORECASE),
    re.compile(r".*?_L(?P<level>\d+)_(?P<row>\d+)_(?P<col>\d+).*?\.(png|webp)$", re.IGNORECASE),
]

@dataclass
class DrawPOI:
    x: float
    y: float
    icon: str
    text: str

def read_csv(path: Path) -> List[dict]:
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

def parse_tile_name(name: str) -> Optional[Tuple[int, int, int]]:
    for pat in TILE_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group("level")), int(m.group("row")), int(m.group("col"))
    return None

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

    files = sorted(list(tile_dir.glob("*.png")) + list(tile_dir.glob("*.webp")))
    if not files:
        raise FileNotFoundError(f"No tiles found in {tile_dir}")

    tiles: Dict[Tuple[int, int], Image.Image] = {}
    max_row = -1
    max_col = -1

    # Prefer L0 tiles when available. Keep row/col placement from filename.
    for path in files:
        parsed = parse_tile_name(path.name)
        if not parsed:
            continue
        level, row, col = parsed
        if level != 0:
            continue
        img = Image.open(path).convert("RGBA")
        tiles[(row, col)] = img
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    # Fallback to a simple sorted-grid stitch if filenames do not match.
    if not tiles:
        opened = [Image.open(p).convert("RGBA") for p in files]
        if not opened:
            raise RuntimeError(f"No usable tiles found in {tile_dir}")
        # Try a square-ish grid. Great Hollow example is 36 tiles -> 6x6.
        count = len(opened)
        side = int(round(count ** 0.5))
        if side * side != count:
            raise RuntimeError(f"Could not parse tile indexes in {tile_dir} and tile count {count} is not a square")
        tile_w, tile_h = opened[0].width, opened[0].height
        canvas = Image.new("RGBA", (tile_w * side, tile_h * side), (0, 0, 0, 0))
        for idx, img in enumerate(opened):
            row = idx // side
            col = idx % side
            canvas.alpha_composite(img, dest=(col * tile_w, row * tile_h))
        return canvas.rotate(90, expand=True)

    sample = next(iter(tiles.values()))
    tile_w, tile_h = sample.width, sample.height
    canvas = Image.new("RGBA", ((max_col + 1) * tile_w, (max_row + 1) * tile_h), (0, 0, 0, 0))
    for (row, col), img in tiles.items():
        canvas.alpha_composite(img, dest=(col * tile_w, row * tile_h))

    # Keep same orientation assumption as the earlier renderer path.
    return canvas.rotate(90, expand=True)

def try_load_font(font_path: Optional[Path], size: int):
    candidates: List[Path] = []
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

def resolve_icon_path(repo: Path, icon_name: str) -> Optional[Path]:
    poi_dir = repo / "public" / "POI_icons"
    path = poi_dir / icon_name
    if path.exists():
        return path
    fallback = MANUAL_ICON_FALLBACKS.get(icon_name)
    if fallback:
        fp = poi_dir / fallback
        if fp.exists():
            return fp
    return None

def get_resized_icon(repo: Path, icon_name: str, cache: Dict[str, Image.Image]) -> Optional[Image.Image]:
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

def strip_category_prefix(text: str) -> str:
    if " - " in text:
        return text.split(" - ", 1)[1].strip()
    return text.strip()

def resolve_smallbase_name(smallbase_map_id: int, variation_id: int) -> str:
    base = KNOWN_SMALLBASE_NAMES.get(smallbase_map_id)
    if base is None:
        return str(smallbase_map_id)

    if ";" not in base:
        return base

    if " - " in base:
        prefix, rest = base.split(" - ", 1)
        options = [p.strip() for p in rest.split(";") if p.strip()]
        if 0 <= variation_id < len(options):
            return f"{prefix} - {options[variation_id]}"
        return base

    options = [p.strip() for p in base.split(";") if p.strip()]
    if 0 <= variation_id < len(options):
        return options[variation_id]
    return base

def determine_shifting_earth(pattern_id: int, flags_by_pattern: Dict[int, List[dict]]) -> str:
    mods = {as_int(r.get("modifier"), -999) for r in flags_by_pattern.get(pattern_id, [])}
    for mod, name in SHIFTING_EARTH_BY_MODIFIER.items():
        if mod in mods:
            return name
    return "Default"

def infer_spawn_modifier(pattern_id: int, flags_by_pattern: Dict[int, List[dict]], sortie_by_modifier: Dict[int, dict]) -> Optional[int]:
    mods = []
    for row in flags_by_pattern.get(pattern_id, []):
        mod = as_int(row.get("modifier"))
        if mod is not None and mod in sortie_by_modifier:
            mods.append(mod)
    # Prefer the last matching modifier if multiple exist.
    return mods[-1] if mods else None

def classify_row(attach_id: int, smallbase_map_id: int, resolved_text: str) -> str:
    if attach_id == 190 or resolved_text.startswith("Castle -"):
        return "Castle"
    if attach_id in {756, 757} or smallbase_map_id in BASEMENT_BOSS_IDS:
        return "Arena Boss"
    if 601 <= attach_id <= 607:
        return "Evergaol"
    if 750 <= attach_id <= 763:
        if smallbase_map_id in FIELD_BOSS_RED_IDS:
            return "Field Boss Red"
        return "Field Boss"
    if smallbase_map_id in EVENT_IDS or resolved_text.startswith("Event -"):
        return "Map Event"
    if 710 <= attach_id <= 718:
        return "Scale-Bearing Merchant"
    if resolved_text.startswith(MAJOR_PREFIXES):
        return "Major Base"
    if resolved_text.startswith(MINOR_PREFIXES):
        return "Minor Base"
    # Fallback by attach ranges
    if (100 <= attach_id < 200) or (1100 <= attach_id < 1300) or (2000 <= attach_id < 2200):
        return "Major Base"
    if (300 <= attach_id < 400) or (2300 <= attach_id < 2400):
        return "Minor Base"
    return "Other"

def infer_icon_and_text(category: str, resolved_text: str) -> Tuple[Optional[str], str]:
    text = strip_category_prefix(resolved_text)

    if category == "Castle":
        return "Castle.png", text
    if category == "Field Boss Red":
        return "Field_Boss_Red.png", text
    if category in {"Field Boss", "Arena Boss"}:
        return "Field_Boss.png", text
    if category == "Evergaol":
        return "Evergaol.png", text
    if category == "Map Event":
        return "Event.png", text
    if category == "Scale-Bearing Merchant":
        return "Scale_Bearing_Merchant.png", "Scale-Bearing Merchant"
    if category == "Major Base":
        if resolved_text.startswith("Fort -"):
            return "Fort.png", text
        if resolved_text.startswith("Camp -"):
            return "Main_Encampment.png", text
        if resolved_text.startswith("Ruins -"):
            return "Ruins.png", text
        if resolved_text.startswith("Great Church -"):
            return "Great_Church.png", text
        if resolved_text.startswith("Castle -"):
            return "Castle.png", text
        return "Great_Church.png", text
    if category == "Minor Base":
        if resolved_text.startswith("Church -"):
            return "Church.png", text
        if resolved_text.startswith("Sorcerer's Rise"):
            return "Sorcerer's_Rise.png", text
        if resolved_text.startswith("Township -"):
            return "Township.png", text
        return "Township.png", text
    return None, text

def build_pois_for_pattern(
    pattern_id: int,
    play_row: dict,
    result_rows_for_pattern: List[dict],
    flags_by_pattern: Dict[int, List[dict]],
    attach_by_id: Dict[int, dict],
    playarea_by_id: Dict[int, dict],
    sortie_by_modifier: Dict[int, dict],
) -> Tuple[str, List[DrawPOI]]:
    shifting_earth = determine_shifting_earth(pattern_id, flags_by_pattern)
    pois: List[DrawPOI] = []

    # Spawn point
    spawn_mod = infer_spawn_modifier(pattern_id, flags_by_pattern, sortie_by_modifier)
    if spawn_mod is not None:
        row = sortie_by_modifier.get(spawn_mod)
        if row:
            gx = as_int(row.get("spawnGridXNo"))
            gz = as_int(row.get("spawnGridZNo"))
            # SortieStartingPoint does not give posX/posZ here, so use tile center.
            if None not in (gx, gz):
                x, y = frontend_xy_from_raw(gx, gz, 0.0, 0.0)
                pois.append(DrawPOI(x, y, "Spawn_Location.png", f"Spawn {spawn_mod}"))

    # Night circles + boss names
    for day, play_field, boss_field in [
        (1, "playArea1", "bossId1"),
        (2, "playArea2", "bossId2"),
    ]:
        play_id = as_int(play_row.get(play_field))
        boss_id = as_int(play_row.get(boss_field))
        if play_id is None:
            continue
        pa = playarea_by_id.get(play_id)
        if not pa:
            continue
        gx = as_int(pa.get("gridXNo"))
        gz = as_int(pa.get("gridZNo"))
        px = as_float(pa.get("posX"))
        pz = as_float(pa.get("posZ"))
        if None in (gx, gz, px, pz):
            continue
        x, y = frontend_xy_from_raw(gx, gz, px, pz)
        boss_text = BOSS_ID_TO_NAME.get(boss_id, str(boss_id) if boss_id is not None else f"Day {day} Boss")
        # Include extra boss on same circle if present.
        extra_field = "extraBossId1" if day == 1 else "extraBossId2"
        extra_id = as_int(play_row.get(extra_field))
        if extra_id is not None and extra_id >= 0:
            extra_text = BOSS_ID_TO_NAME.get(extra_id, str(extra_id))
            text = f"Night {day}\n{boss_text}\nExtra: {extra_text}"
        else:
            text = f"Night {day}\n{boss_text}"
        pois.append(DrawPOI(x, y, "Night_Location.png", text))

    # POIs from LotResultSmallBaseAndSpot
    for row in result_rows_for_pattern:
        attach_id = as_int(row.get("attachId"))
        smallbase_map_id = as_int(row.get("smallBaseMapId"))
        variation_id = as_int(row.get("variationId"), 0)
        if None in (attach_id, smallbase_map_id):
            continue

        attach = attach_by_id.get(attach_id)
        if not attach:
            continue
        gx = as_int(attach.get("gridXNo"))
        gz = as_int(attach.get("gridZNo"))
        px = as_float(attach.get("posX"))
        pz = as_float(attach.get("posZ"))
        if None in (gx, gz, px, pz):
            continue

        resolved_text = resolve_smallbase_name(smallbase_map_id, variation_id)
        category = classify_row(attach_id, smallbase_map_id, resolved_text)
        icon, text = infer_icon_and_text(category, resolved_text)
        if not icon:
            continue
        x, y = frontend_xy_from_raw(gx, gz, px, pz)
        pois.append(DrawPOI(x, y, icon, text))

    return shifting_earth, pois

def render_pattern(
    repo: Path,
    game_csv_dir: Path,
    pattern_id: int,
    show_text: bool,
    out_path: Path,
    font_path: Optional[Path],
    play_rows_by_id: Dict[int, dict],
    results_by_pattern: Dict[int, List[dict]],
    flags_by_pattern: Dict[int, List[dict]],
    attach_by_id: Dict[int, dict],
    playarea_by_id: Dict[int, dict],
    sortie_by_modifier: Dict[int, dict],
) -> None:
    play_row = play_rows_by_id.get(pattern_id)
    if play_row is None:
        raise KeyError(f"No LotResultPlayAreaParam row with ID={pattern_id}")

    shifting_earth, pois = build_pois_for_pattern(
        pattern_id=pattern_id,
        play_row=play_row,
        result_rows_for_pattern=results_by_pattern.get(pattern_id, []),
        flags_by_pattern=flags_by_pattern,
        attach_by_id=attach_by_id,
        playarea_by_id=playarea_by_id,
        sortie_by_modifier=sortie_by_modifier,
    )
    map_layout = map_layout_from_shifting_earth(shifting_earth)
    print(f"[pattern {pattern_id}] shifting_earth={shifting_earth} map_layout={map_layout} pois={len(pois)}")

    base = stitch_map_tiles(repo, map_layout)
    draw = ImageDraw.Draw(base)
    font = try_load_font(font_path, TITLE_FONT_SIZE)
    icon_cache: Dict[str, Image.Image] = {}

    for poi in pois:
        icon = get_resized_icon(repo, poi.icon, icon_cache)
        if icon is None:
            print(f"Warning: missing icon {poi.icon}")
            continue

        # Existing frontend coordinates assume 2204x1690 reference space.
        sx = poi.x / 2204.0 * base.width
        sy = poi.y / 1690.0 * base.height

        base.alpha_composite(icon, dest=(int(round(sx - icon.width / 2)), int(round(sy - icon.height / 2))))
        if show_text and poi.text:
            draw.multiline_text(
                (sx, sy + icon.height / 2 + 10),
                poi.text,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=TITLE_STROKE,
                anchor="ma",
                align="center",
                spacing=3,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Respect extension chosen by caller.
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        base.convert("RGB").save(out_path, quality=95)
    elif out_path.suffix.lower() == ".webp":
        base.save(out_path, format="WEBP", quality=95, method=6)
    else:
        base.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Render Nightreign patterns directly from game CSVs (no layout_XXX.json needed).")
    ap.add_argument("--repo", type=Path, required=True, help="Repo root containing public/assets/maps and public/POI_icons")
    ap.add_argument("--game-csv-dir", type=Path, required=True, help="Folder containing exported game CSVs")
    ap.add_argument("--pattern-id", type=int, nargs="+", required=True, help="LotResultPlayAreaParam.ID values to render")
    ap.add_argument("--mode", choices=["with_text", "no_text", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--font", type=Path, default=None)
    ap.add_argument("--ext", choices=["png", "jpg", "webp"], default="png")
    args = ap.parse_args()

    game_csv_dir = args.game_csv_dir
    lot_result_play = read_csv(game_csv_dir / "LotResultPlayAreaParam.csv")
    lot_result_small = read_csv(game_csv_dir / "LotResultSmallBaseAndSpot.csv")
    lot_result_flags = read_csv(game_csv_dir / "LotResultMapPatternFlag.csv")
    attach_rows = read_csv(game_csv_dir / "SmallBaseAndSpotAttachPoint.csv")
    playarea_rows = read_csv(game_csv_dir / "PlayAreaCreateParam.csv")
    sortie_rows = read_csv(game_csv_dir / "SortieStartingPoint.csv")

    play_rows_by_id = {as_int(r.get("ID")): r for r in lot_result_play if as_int(r.get("ID")) is not None}
    results_by_pattern: Dict[int, List[dict]] = defaultdict(list)
    for r in lot_result_small:
        pid = as_int(r.get("patternId"))
        if pid is not None:
            results_by_pattern[pid].append(r)

    flags_by_pattern: Dict[int, List[dict]] = defaultdict(list)
    for r in lot_result_flags:
        pid = as_int(r.get("patternId"))
        if pid is not None:
            flags_by_pattern[pid].append(r)

    attach_by_id = {as_int(r.get("ID")): r for r in attach_rows if as_int(r.get("ID")) is not None}
    playarea_by_id = {as_int(r.get("ID")): r for r in playarea_rows if as_int(r.get("ID")) is not None}
    sortie_by_modifier = {as_int(r.get("requireModifier")): r for r in sortie_rows if as_int(r.get("requireModifier")) is not None}

    for pattern_id in args.pattern_id:
        if args.mode == "both":
            modes = [("no_text", False), ("with_text", True)]
        elif args.mode == "with_text":
            modes = [("with_text", True)]
        else:
            modes = [("no_text", False)]

        for label, show_text in modes:
            out_path = args.out_dir / f"pattern_{pattern_id}_{label}.{args.ext}"
            render_pattern(
                repo=args.repo,
                game_csv_dir=game_csv_dir,
                pattern_id=pattern_id,
                show_text=show_text,
                out_path=out_path,
                font_path=args.font,
                play_rows_by_id=play_rows_by_id,
                results_by_pattern=results_by_pattern,
                flags_by_pattern=flags_by_pattern,
                attach_by_id=attach_by_id,
                playarea_by_id=playarea_by_id,
                sortie_by_modifier=sortie_by_modifier,
            )

if __name__ == "__main__":
    main()
