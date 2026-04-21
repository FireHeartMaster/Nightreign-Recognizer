#!/usr/bin/env python3
# v11 note:
# - keeps the validated hard-coded Castle render offset for POI 159
# - preserves the Arena_Boss label fix from v9
# - restores the Castle enemy-type text label (e.g. "Trolls") for POI 159
#   and starts that title from the same offset-adjusted position as the Castle icon
# Render Elden Ring: Nightreign map images from the Nightreign Router repo,
# with or without text labels.
#
# v9 fix: treat Arena Boss POIs like Field Bosses for label generation and
# metadata collection so bosses such as Grafted Scion still get titles even
# when Arena_Boss.png falls back to Field_Boss.png.
#
# This script mirrors the main parts of the router's map pipeline:
# - stitch the 6x6 surface tiles using the same 90° counterclockwise rotation
# - load dynamic POIs from layout_XXX.json
# - load fixed POIs from the map-specific coordinate layout JSON
# - prefer dynamic POIs when a dynamic and fixed POI share the same coordinate
# - apply the same key layout-specific filters/special cases used by MapCanvas.tsx
# - optionally draw text labels with collision avoidance similar to MapCanvas.tsx
#
# Typical usage:
#   python render_nightreign_router_map.py \
#       --layout 35 \
#       --out gladius_035_no_text.png \
#       --text off
#
#   python render_nightreign_router_map.py \
#       --layout 35 \
#       --out gladius_035_with_text.png \
#       --text on
#
#   python render_nightreign_router_map.py \
#       --filter-shifting-earth "Rotted Woods" \
#       --mode both \
#       --out-dir rotted_woods_renders \
#       --name-mode numbered \
#       --zip


from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TextIO

from PIL import Image, ImageDraw, ImageFont


def runtime_repo_root() -> Path:
    return Path(__file__).resolve().parent

# Public repo fallback: poiIconMapping.ts references Arena_Boss.png,
# but the current public public/POI_icons folder does not contain that file.
# This map keeps rendering working until/if the real icon is added.
MANUAL_ICON_FALLBACKS = {
    'Map_Event.png': 'Event.png',
    'Arena_Boss.png': 'Field_Boss.png',
}

DEBUG_CASTLE_ENABLED = False
DEBUG_CASTLE_STREAM: Optional[TextIO] = None


LEFT_BOUND = 507.0
ACTIVE_WIDTH = 1690.0
RAW_MAP_HEIGHT = 1690.0
GRID_SIZE = 6
TITLE_FONT_SIZE = 21
TITLE_STROKE = 1
TITLE_MAX_MOVEMENT = 48.0

# Small visual anchor correction for the Castle icon.
# Adjust these if the Castle still appears slightly off.
CASTLE_RENDER_OFFSET_X = 50.0
CASTLE_RENDER_OFFSET_Y = 75.0


POI_TYPE_ICON_MAP = {
    "castle": "Field_Boss.png",
    "Castle": "Field_Boss.png",
    "Buried_Treasures": "Buried_Treasure.png",
    "Tunnel_Entrances": "Tunnel_Entrance.png",
    "Spiritstreams": "Spiritstream.png",
    "Spectral_Hawk_Trees": "Spectral_Hawk_Tree.png",
    "Smithing_Tables": "Ruins.png",
    "Sites_of_grace": "Site_of_Grace.png",
    "Scarabs": "Scarab.png",
    "Minor_Locations": "Township.png",
    "Merchants": "Main_Encampment.png",
    "Major_Locations": "Great_Church.png",
    "Field_Bosses": "Field_Boss.png",
    "Evergaols": "Evergaol.png",
    "Mountaintops_Buff": "Mission_Objective.png",
    "Rot_Blessing": "Mission_Objective.png",
    "Event_Locations": "Event.png",
    "Circle_Locations": "Night_Location.png",
    "Scale_Bearing_Merchant_Locations": "Scale_Bearing_Merchant.png",
    "Spawn_Locations": "Spawn_Location.png",
}

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
    "Buried_Treasure.png": {"width": 48},
    "Tunnel_Entrance.png": {"width": 48},
    "Township.png": {"width": 96},
    "Spiritstream.png": {"width": 32},
    "Spectral_Hawk_Tree.png": {"width": 48},
    "Sorcerer's_Rise.png": {"width": 48},
    "Site_of_Grace.png": {"width": 48},
    "Scarab.png": {"width": 32},
    "Ruins.png": {"width": 96},
    "Event.png": {"width": 96},
    "Night_Location.png": {"width": 96},
    "Scale_Bearing_Merchant.png": {"width": 48},
    "Spawn_Location.png": {"width": 128},
    "Map_Event.png": {"width": 96},
}

ROT_BLESSING_POI_MAP = {
    "Southwest": 164,
    "Northeast": 165,
    "West": 166,
}

FRENZY_TOWER_MAPPING = {
    "South": 196,
    "North": 199,
}

MOUNTAINTOPS_HARDCODED_TITLES = {
    140: "Flying Dragon",
    141: "Mountaintop Ice Dragon",
    142: "Mountaintop Ice Dragon",
    143: "Demi-Human Swordmaster",
    144: "Giant Crows",
    145: "Mountaintop Ice Dragon",
    146: "Demi-Human Swordmaster",
    147: "Mountaintop Ice Dragon",
    148: "Snowfield Trolls",
    149: "Albinauric Archers",
}

CRATER_HARDCODED_TITLES = {
    132: "Red Wolf",
    133: "Demi-Human Queen",
    134: "Fire Prelates",
    135: "Demi-Human Queen",
    136: "Magma Wyrm",
    137: "Fallingstar Beast",
    138: "Valiant Gargoyle",
    91: "Flying Dragon",
}

NOKLATEO_HARDCODED_TITLES = {
    123: "Royal Carian Knight",
    124: "Flying Dragon",
    125: "Black Knife Assassin",
    126: "Headless Troll",
    127: "Dragonkin Soldier",
    128: "Royal Revenant",
    129: "Black Knife Assassin",
    130: "Astel",
    23: "Golden Hippopotamus",
}

ROTTEN_WOODS_HARDCODED_TITLES = {
    156: "Lordsworn Captain",
}

RECOGNITION_PACKAGE_VERSION = 4
RECOGNITION_CMP_SIZE = 512
RECOGNITION_ICON_CANVAS_SIZE = 192
LANDMARK_ICON_TO_TYPE = {
    "Church.png": "Church",
    "Fort.png": "Fort",
    "Great_Church.png": "Great Church",
    "Main_Encampment.png": "Main Encampment",
    "Ruins.png": "Ruins",
    "Sorcerer's_Rise.png": "Sorcerer's Rise",
}


@dataclass
class POI:
    id: int
    x: float
    y: float
    poi_type: str
    icon: Optional[str] = None
    value: Optional[str] = None


@dataclass
class DynamicData:
    layout_number: int
    shifting_earth: str
    map_layout: str
    dynamic_pois: List[POI]
    castle_enemy_type: Optional[str]
    evergaol_bosses: List[Dict[str, object]]
    field_bosses: List[Dict[str, object]]
    major_locations: List[Dict[str, object]]
    night_circles: List[Dict[str, object]]
    sorcerers_rise_locations: List[Dict[str, object]]
    layout_data: Dict[str, object]


@dataclass
class TitlePlacement:
    id: int
    x: float
    y: float
    text: str
    priority: int


@dataclass
class RenderMode:
    name: str
    show_text: bool


def debug_castle(message: str) -> None:
    if not DEBUG_CASTLE_ENABLED:
        return
    line = f"[castle-debug] {message}"
    print(line)
    if DEBUG_CASTLE_STREAM is not None:
        DEBUG_CASTLE_STREAM.write(line + "\n")
        DEBUG_CASTLE_STREAM.flush()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Nightreign Router maps with or without labels.")
    parser.add_argument("--repo", type=Path, default=None, help="Optional override for the repository root. Defaults to the folder containing this script.")
    parser.add_argument("--layout", type=int, nargs="*", default=[], help="Specific layout number(s) to render")
    parser.add_argument("--filter-shifting-earth", type=str, default=None, help='Only render layouts whose "Shifting Earth" matches this string, e.g. "Rotted Woods"')
    parser.add_argument("--mode", choices=["with_text", "no_text", "both"], default="both", help="What to render")
    parser.add_argument("--text", choices=["on", "off"], default=None, help="Convenience alias for single-image use; overrides --mode")
    parser.add_argument("--out", type=Path, default=None, help="Single output file; only valid when rendering exactly one image")
    parser.add_argument("--out-dir", type=Path, default=Path("renders"), help="Output directory for batch rendering")
    parser.add_argument("--name-mode", choices=["layout", "numbered"], default="layout", help="Batch file naming scheme")
    parser.add_argument("--start-number", type=int, default=1, help="Starting number when --name-mode numbered")
    parser.add_argument("--ext", choices=["png", "jpg", "webp"], default="png", help="Output extension")
    parser.add_argument("--zip", action="store_true", help="Create zip file(s) after batch rendering")
    parser.add_argument("--font", type=Path, default=None, help="Optional TTF/OTF font path for labels")
    parser.add_argument("--hide-field-bosses", action="store_true", help="Do not render Field_Boss or Field_Boss_Red icons.")
    parser.add_argument("--hide-night-bosses", action="store_true", help="Do not render Night_Location icons for day 1/day 2 bosses.")
    parser.add_argument("--hide-spawn-location", action="store_true", help="Do not render the Spawn_Location icon.")
    parser.add_argument("--castle-offset-x", type=float, default=CASTLE_RENDER_OFFSET_X, help="Extra pixel offset applied to the Castle icon on the X axis after map scaling.")
    parser.add_argument("--castle-offset-y", type=float, default=CASTLE_RENDER_OFFSET_Y, help="Extra pixel offset applied to the Castle icon on the Y axis after map scaling.")
    parser.add_argument("--debug-castle", action="store_true", help="Print detailed logs about Castle (POI id 159) resolution, dedupe, and rendering.")
    parser.add_argument("--debug-castle-log", type=Path, default=None, help="Optional file to also write castle debug logs to.")
    parser.add_argument("--export-recognition-package", type=Path, default=None, help="Optional .npz package for the geometric prototype recognizer.")
    parser.add_argument("--export-recognition-dir", type=Path, default=None, help="Optional directory to write per-layout landmark metadata JSON files for the geometric prototype recognizer.")
    return parser.parse_args()


def determine_modes(args: argparse.Namespace) -> List[RenderMode]:
    if args.text is not None:
        return [RenderMode("with_text" if args.text == "on" else "no_text", args.text == "on")]
    if args.mode == "with_text":
        return [RenderMode("with_text", True)]
    if args.mode == "no_text":
        return [RenderMode("no_text", False)]
    return [RenderMode("no_text", False), RenderMode("with_text", True)]


def map_layout_from_shifting_earth(shifting_earth: str) -> str:
    return {
        "Mountaintop": "the_mountaintop_shifted",
        "Crater": "the_crater_shifted",
        "Rotted Woods": "the_rotten_woods_shifted",
        "Noklateo": "noklateo_shifted",
        "Great Hollow": "great_hollow_shifted",
        "Default": "default",
    }.get(shifting_earth, "default")


def parse_value_icon_mapping(ts_text: str) -> Dict[str, str]:
    pattern = re.compile(r'\{\s*value:\s*"([^"]+)",\s*icon:\s*"([^"]+)"', re.MULTILINE)
    return {value: icon for value, icon in pattern.findall(ts_text)}


def parse_location_mapping(ts_text: str) -> Dict[str, int]:
    pattern = re.compile(r'\{\s*layoutLocation:\s*"([^"]+)",\s*poiId:\s*(\d+)', re.MULTILINE)
    mapping: Dict[str, int] = {}
    for location, poi_id in pattern.findall(ts_text):
        mapping[location] = int(poi_id)
    return mapping


def build_context_mappings(location_ts: str) -> Dict[str, Dict[str, int]]:
    spawn = {
        "Above Stormhill Tunnel Entrance": 202,
        "Below Summonwater Hawk": 204,
        "East of Cavalry Bridge": 201,
        "Far Southwest": 207,
        "Minor Erdtree": 206,
        "Northeast of Saintsbridge": 208,
        "Southeast of Lake": 205,
        "Stormhill South of Gate": 203,
        "West of Warmaster's Shack": 209,
    }
    night_circle = {
        "East of Saintsbridge": 182,
        "Northeast Corner": 177,
        "Northeast of Lake": 184,
        "Northwest Corner": 178,
        "Northwest Lake": 185,
        "Northwest Mistwood Pond": 181,
        "Northwest of Castle": 183,
        "South Lake": 179,
        "South of Castle": 187,
        "Southwest Corner": 180,
        "Southwest Mistwood": 186,
        "West Stormhill Graveyard": 176,
        "North of Crater": 211,
        "Southeast Mountaintop": 212,
        "Noklateo Entrance": 210,
        "Southeast Rotted Woods": 213,
        "Northwest Rotted Woods": 214,
        "South Lake Night Circle": 179,
        "South of Castle Night Circle": 187,
    }
    merchant = {
        "South of Castle": 169,
        "Ruins North of Castle": 170,
        "Northeast Corner": 171,
        "Southeast Lakeshore": 172,
        "Northwest Corner": 27,
        "East of Saintsbridge": 173,
        "Castle Front": 174,
        "Southwest Corner": 175,
    }
    return {"spawn": spawn, "night_circle": night_circle, "merchant": merchant}


def repo_paths(repo: Path) -> Dict[str, Path]:
    return {
        "layouts": repo / "reference_material" / "pattern_layouts",
        "master_poi": repo / "public" / "assets" / "maps" / "poi_coordinates_with_ids.json",
        "coord_dir": repo / "public" / "assets" / "maps" / "coordinates",
        "map_dir": repo / "public" / "assets" / "maps",
        "icon_dir": repo / "public" / "POI_icons",
        "map_canvas": repo / "src" / "app" / "_components" / "MapCanvas.tsx",
        "icon_mapping": repo / "src" / "utils" / "poiIconMapping.ts",
        "location_mapping": repo / "src" / "utils" / "poiLocationMapping.ts",
    }


def validate_repo(repo: Path) -> Dict[str, Path]:
    paths = repo_paths(repo)
    required = [
        paths["layouts"],
        paths["master_poi"],
        paths["coord_dir"],
        paths["icon_dir"],
        paths["map_canvas"],
        paths["icon_mapping"],
        paths["location_mapping"],
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The repo path does not look like a complete nightreign-map-router checkout. Missing:\n- "
            + "\n- ".join(missing)
        )
    return paths


def load_master_poi(master_poi_path: Path) -> Dict[int, Tuple[float, float]]:
    data = load_json(master_poi_path)
    return {int(item["id"]): (float(item["coordinates"][0]), float(item["coordinates"][1])) for item in data}


def get_poi_id_for_location_with_context(key: str, location: str, location_map: Dict[str, int]) -> Optional[int]:
    if location == "Minor Erdtree":
        key_type = key.split(" - ")[0]
        if key_type == "Major Base":
            return 106
        if key_type == "Minor Base":
            return 157
    return location_map.get(location)


def get_icon_for_poi_with_context(key: str, value: str, value_icon_map: Dict[str, str], special_event: Optional[str]) -> Optional[str]:
    key_type = key.split(" - ")[0]

    if value == "Dragonkin Soldier":
        if key_type == "Evergaol":
            return "Evergaol.png"
        if key_type in {"Field Boss", "Night"}:
            return "Field_Boss.png"

    if special_event:
        is_meteor = "Meteor Strike" in special_event
        is_mausoleum = "Walking Mausoleum" in special_event
        if (is_meteor or is_mausoleum) and value.startswith("Map Event -"):
            return "Event.png"

    return value_icon_map.get(value)


def load_dynamic_data(
    layout_number: int,
    layout_data: Dict[str, object],
    master_poi: Dict[int, Tuple[float, float]],
    value_icon_map: Dict[str, str],
    location_map: Dict[str, int],
    context_maps: Dict[str, Dict[str, int]],
) -> DynamicData:
    dynamic_pois: List[POI] = []
    castle_enemy_type: Optional[str] = None
    evergaol_bosses: List[Dict[str, object]] = []
    field_bosses: List[Dict[str, object]] = []
    major_locations: List[Dict[str, object]] = []
    night_circles: List[Dict[str, object]] = []
    sorcerers_rise_locations: List[Dict[str, object]] = []

    castle_obj = layout_data.get("Castle - Castle")
    if isinstance(castle_obj, dict):
        castle_enemy_type = castle_obj.get("value")

    special_event = layout_data.get("Special Event")
    special_event = special_event if isinstance(special_event, str) else None

    for key, obj in layout_data.items():
        if isinstance(obj, dict) and "location" in obj and "value" in obj:
            location = obj.get("location")
            poi_value = obj.get("value")
            if not isinstance(location, str) or not isinstance(poi_value, str) or poi_value == "empty":
                continue

            poi_id = get_poi_id_for_location_with_context(key, location, location_map)
            if poi_id is None or poi_id not in master_poi:
                continue

            icon = get_icon_for_poi_with_context(key, poi_value, value_icon_map, special_event)
            if not icon:
                continue

            x, y = master_poi[poi_id]
            poi = POI(id=poi_id, x=x, y=y, poi_type=key.split(" - ")[0], icon=icon, value=poi_value)
            dynamic_pois.append(poi)

            if icon == "Evergaol.png":
                evergaol_bosses.append({"id": poi_id, "boss": poi_value})
            if icon in {"Field_Boss.png", "Field_Boss_Red.png", "Arena_Boss.png"}:
                field_bosses.append({"id": poi_id, "boss": poi_value})
            if key.startswith("Major Base -"):
                clean_value = re.sub(r'^(Ruins|Camp|Fort|Great Church|Church|Sorcerer\'s Rise|Township) - ', "", poi_value)
                major_locations.append({"id": poi_id, "location": clean_value})
            if poi_value.startswith("Sorcerer's Rise -"):
                clean_value = re.sub(r"^Sorcerer's Rise - ", "", poi_value)
                sorcerers_rise_locations.append({"id": poi_id, "location": clean_value})

    spawn_point = layout_data.get("Spawn Point")
    if isinstance(spawn_point, str) and spawn_point != "empty":
        poi_id = context_maps["spawn"].get(spawn_point)
        if poi_id in master_poi:
            x, y = master_poi[poi_id]
            dynamic_pois.append(POI(poi_id, x, y, "Spawn Point", "Spawn_Location.png", "Spawn Point"))

    for night_num in (1, 2):
        circle_key = f"Night {night_num} Circle"
        boss_key = f"Night {night_num} Boss"
        location = layout_data.get(circle_key)
        if isinstance(location, str) and location != "empty":
            poi_id = context_maps["night_circle"].get(location)
            if poi_id in master_poi:
                boss = layout_data.get(boss_key) or "Night Boss"
                if not isinstance(boss, str):
                    boss = "Night Boss"
                x, y = master_poi[poi_id]
                dynamic_pois.append(POI(poi_id, x, y, "Night Circle", "Night_Location.png", boss))
                night_circles.append({"id": poi_id, "boss": boss, "night": night_num})

    merchant = layout_data.get("Scale-Bearing Merchant")
    if isinstance(merchant, str) and merchant != "empty":
        poi_id = context_maps["merchant"].get(merchant)
        if poi_id in master_poi:
            x, y = master_poi[poi_id]
            dynamic_pois.append(POI(poi_id, x, y, "Merchant", "Scale_Bearing_Merchant.png", "Scale-Bearing Merchant"))

    frenzy_tower = layout_data.get("Frenzy Tower")
    if isinstance(frenzy_tower, str) and frenzy_tower != "empty":
        poi_id = FRENZY_TOWER_MAPPING.get(frenzy_tower)
        if poi_id in master_poi:
            x, y = master_poi[poi_id]
            dynamic_pois.append(POI(poi_id, x, y, "Frenzy Tower", "Event.png", "Frenzy Tower"))

    shifting_earth = layout_data.get("Shifting Earth") or "Default"
    if not isinstance(shifting_earth, str):
        shifting_earth = "Default"

    return DynamicData(
        layout_number=layout_number,
        shifting_earth=shifting_earth,
        map_layout=map_layout_from_shifting_earth(shifting_earth),
        dynamic_pois=dynamic_pois,
        castle_enemy_type=castle_enemy_type,
        evergaol_bosses=evergaol_bosses,
        field_bosses=field_bosses,
        major_locations=major_locations,
        night_circles=night_circles,
        sorcerers_rise_locations=sorcerers_rise_locations,
        layout_data=layout_data,
    )


def load_coordinate_layout(coord_dir: Path, map_layout: str) -> Dict[str, List[List[float]]]:
    # Match Nightreign Router's getCoordinateJsonPath() naming.
    coord_name = {
        "noklateo_shifted": "noklateo_map_layout.json",
        "the_crater_shifted": "the_crater_map_layout.json",
        "the_mountaintop_shifted": "the_mountaintop_map_layout.json",
        "the_rotten_woods_shifted": "the_rotten_woods_map_layout.json",
        "great_hollow_shifted": "great_hollow_map_layout.json",
        "default": "default_map_layout.json",
    }.get(map_layout, "default_map_layout.json")

    coord_path = coord_dir / coord_name
    if not coord_path.exists() and map_layout != "default":
        coord_path = coord_dir / "default_map_layout.json"
    return load_json(coord_path)


def find_master_poi_by_coord(master_poi: Dict[int, Tuple[float, float]], x: float, y: float, epsilon: float = 0.01) -> Optional[int]:
    for poi_id, (px, py) in master_poi.items():
        if abs(px - x) < epsilon and abs(py - y) < epsilon:
            return poi_id
    return None


def build_all_pois(
    coord_layout: Dict[str, List[List[float]]],
    master_poi: Dict[int, Tuple[float, float]],
    dynamic_data: DynamicData,
) -> List[POI]:
    all_pois: List[POI] = list(dynamic_data.dynamic_pois)

    for poi in dynamic_data.dynamic_pois:
        if poi.id == 159:
            debug_castle(
                f"dynamic POI id=159 added before fixed merge: type={poi.poi_type}, "
                f"icon={poi.icon}, value={poi.value}, x={poi.x}, y={poi.y}"
            )

    unique_fixed: Dict[str, POI] = {}
    for poi_type, coords in coord_layout.items():
        if poi_type in {"Spawn_Locations", "Event_Locations", "Scale_Bearing_Merchant_Locations", "Circle_Locations", "Merchants", "Smithing_Tables"}:
            continue

        if poi_type == "Minor_Locations":
            if dynamic_data.map_layout == "the_crater_shifted":
                for x, y in coords:
                    poi_id = find_master_poi_by_coord(master_poi, float(x), float(y))
                    if poi_id == 131:
                        px, py = master_poi[poi_id]
                        key = f"{px},{py}"
                        if key not in unique_fixed:
                            unique_fixed[key] = POI(poi_id, px, py, poi_type)
            continue

        for x, y in coords:
            poi_id = find_master_poi_by_coord(master_poi, float(x), float(y))
            if poi_id is None:
                continue
            px, py = master_poi[poi_id]
            key = f"{px},{py}"
            if key not in unique_fixed:
                unique_fixed[key] = POI(poi_id, px, py, poi_type)
                if poi_id == 159:
                    debug_castle(
                        f"fixed POI id=159 created from coord layout: coord_type={poi_type}, "
                        f"source_x={float(x)}, source_y={float(y)}, master_x={px}, master_y={py}"
                    )

    filtered = [poi for poi in unique_fixed.values() if poi.id not in {24, 25, 26}]

    if dynamic_data.map_layout == "the_crater_shifted":
        filtered = [poi for poi in filtered if poi.id != 88]
    if dynamic_data.map_layout == "the_rotten_woods_shifted":
        filtered = [poi for poi in filtered if poi.id != 92]

    if dynamic_data.map_layout == "noklateo_shifted" and not any(poi.id == 23 for poi in filtered):
        if 23 in master_poi:
            x, y = master_poi[23]
            filtered.append(POI(23, x, y, "Field_Bosses"))

    all_pois.extend(filtered)

    deduped: Dict[str, POI] = {}
    for poi in all_pois:
        key = f"{poi.x},{poi.y}"
        existing = deduped.get(key)

        if existing is not None and (existing.id == 159 or poi.id == 159):
            debug_castle(
                f"dedupe collision at key={key}: existing(id={existing.id}, type={existing.poi_type}, icon={existing.icon}, value={existing.value}) "
                f"vs incoming(id={poi.id}, type={poi.poi_type}, icon={poi.icon}, value={poi.value})"
            )

        if existing is None:
            deduped[key] = poi
            continue

        # Protect the real castle POI from being overwritten by another POI at the same coordinate.
        if existing.id == 159 and poi.id != 159:
            debug_castle(
                f"kept existing castle at key={key}; rejected incoming id={poi.id}, type={poi.poi_type}, icon={poi.icon}, value={poi.value}"
            )
            continue

        if poi.id == 159 and existing.id != 159:
            deduped[key] = poi
            debug_castle(
                f"replaced existing id={existing.id} with castle at key={key}; incoming type={poi.poi_type}, icon={poi.icon}, value={poi.value}"
            )
            continue

        if poi.icon:
            deduped[key] = poi

    final_pois = list(deduped.values())

    if dynamic_data.map_layout == "the_rotten_woods_shifted":
        field_boss_lookup = {int(b["id"]): str(b["boss"]) for b in dynamic_data.field_bosses}
        final_pois = [
            poi for poi in final_pois
            if not (
                (poi.poi_type == "Field_Bosses" or poi.icon in {"Field_Boss.png", "Field_Boss_Red.png"})
                and field_boss_lookup.get(poi.id) == "Putrid Ancestral Followers"
            )
        ]

        rot_blessing_location = dynamic_data.layout_data.get("Rot Blessing")
        if isinstance(rot_blessing_location, str):
            target_id = ROT_BLESSING_POI_MAP.get(rot_blessing_location)
            if target_id:
                final_pois = [poi for poi in final_pois if (poi.id not in {164, 165, 166} or poi.id == target_id)]

    if dynamic_data.layout_data.get("Special Event") == "empty":
        final_pois = [poi for poi in final_pois if not (poi.value and poi.value.startswith("Map Event -"))]

    if any(poi.id == 159 for poi in final_pois):
        castle_poi = next(poi for poi in final_pois if poi.id == 159)
        debug_castle(
            f"final castle kept: id=159, type={castle_poi.poi_type}, icon={castle_poi.icon}, "
            f"value={castle_poi.value}, x={castle_poi.x}, y={castle_poi.y}"
        )
    else:
        debug_castle("final castle missing after POI build and dedupe")

    return final_pois


def format_boss_name(name: str) -> str:
    words = name.split()
    if len(words) <= 1:
        return name
    mid = math.ceil(len(words) / 2)
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def scaled_xy(x: float, y: float, map_width: int, map_height: int, map_layout: str, poi_id: int) -> Tuple[float, float]:
    sx = ((x - LEFT_BOUND) / ACTIVE_WIDTH) * map_width
    sy = (y / RAW_MAP_HEIGHT) * map_height
    if poi_id == 23 and map_layout == "noklateo_shifted":
        sx += 40
    if poi_id == 84:
        sx -= 20
    return sx, sy


def icon_for_poi(poi: POI, map_layout: str) -> Optional[str]:
    icon_file = poi.icon or POI_TYPE_ICON_MAP.get(poi.poi_type) or POI_TYPE_ICON_MAP.get("Default")

    if poi.id == 159:
        icon_file = "Castle.png"

    if poi.id == 149 and poi.poi_type == "Major_Locations":
        icon_file = "Ruins.png"
    if poi.id == 131 and poi.poi_type == "Minor_Locations" and map_layout == "the_crater_shifted":
        icon_file = "Church.png"
    if poi.id == 91 and map_layout == "the_crater_shifted":
        icon_file = "Field_Boss_Red.png"
    if poi.id == 23 and map_layout == "noklateo_shifted":
        icon_file = "Field_Boss.png"
    if poi.id == 156 and map_layout == "the_rotten_woods_shifted":
        icon_file = "Fort.png"
    if poi.id in {141, 142, 145, 147} and map_layout == "the_mountaintop_shifted":
        icon_file = "Field_Boss_Red.png"
    if poi.id in {136, 137, 138} and map_layout == "the_crater_shifted":
        icon_file = "Field_Boss_Red.png"
    if poi.id == 130 and map_layout == "noklateo_shifted":
        icon_file = "Field_Boss_Red.png"
    if poi.id in {123, 127} and map_layout == "noklateo_shifted":
        icon_file = "Field_Boss_Red.png"
    return icon_file


def title_text_for_poi(poi: POI, dynamic_data: DynamicData, icon_file: Optional[str]) -> Optional[Tuple[str, int]]:
    map_layout = dynamic_data.map_layout
    if poi.id == 159:
        if dynamic_data.castle_enemy_type:
            return format_boss_name(str(dynamic_data.castle_enemy_type)), 2
        return None

    if map_layout == "the_mountaintop_shifted" and poi.id in MOUNTAINTOPS_HARDCODED_TITLES:
        return format_boss_name(MOUNTAINTOPS_HARDCODED_TITLES[poi.id]), 2
    if map_layout == "the_crater_shifted" and poi.id in CRATER_HARDCODED_TITLES:
        return format_boss_name(CRATER_HARDCODED_TITLES[poi.id]), 2
    if map_layout == "noklateo_shifted" and poi.id in NOKLATEO_HARDCODED_TITLES:
        return format_boss_name(NOKLATEO_HARDCODED_TITLES[poi.id]), 2
    if map_layout == "the_rotten_woods_shifted" and poi.id in ROTTEN_WOODS_HARDCODED_TITLES:
        return format_boss_name(ROTTEN_WOODS_HARDCODED_TITLES[poi.id]), 2

    if icon_file == "Evergaol.png":
        found = next((b for b in dynamic_data.evergaol_bosses if int(b["id"]) == poi.id), None)
        if found:
            return format_boss_name(str(found["boss"])), 1

    if icon_file in {"Field_Boss.png", "Field_Boss_Red.png", "Arena_Boss.png"} or poi.poi_type in {"Field_Bosses", "Arena Boss"}:
        found = next((b for b in dynamic_data.field_bosses if int(b["id"]) == poi.id), None)
        if found:
            return format_boss_name(str(found["boss"])), 2

    if icon_file in {"Ruins.png", "Main_Encampment.png", "Great_Church.png", "Fort.png", "Church.png", "Township.png"}:
        found = next((m for m in dynamic_data.major_locations if int(m["id"]) == poi.id), None)
        if found:
            return format_boss_name(str(found["location"])), 3

    if icon_file == "Night_Location.png":
        found = next((n for n in dynamic_data.night_circles if int(n["id"]) == poi.id), None)
        if found:
            return f"Night {found['night']}\n{format_boss_name(str(found['boss']))}", 0

    if icon_file == "Sorcerer's_Rise.png":
        found = next((s for s in dynamic_data.sorcerers_rise_locations if int(s["id"]) == poi.id), None)
        if found:
            return format_boss_name(str(found["location"])), 4

    if icon_file == "Event.png":
        special_event = dynamic_data.layout_data.get("Special Event")
        if isinstance(special_event, str) and special_event != "empty":
            m = re.match(r"Day (\d+) (.+)", special_event)
            if m:
                day, event_type = m.groups()
                if event_type == "Frenzy Tower":
                    return "Frenzy Tower", 1
                return f"Day {day}\n{event_type}", 1
            if special_event == "Walking Mausoleum":
                return "Walking Mausoleum", 1

    return None


def get_text_bounds(x: float, y: float, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Tuple[float, float, float, float]:
    lines = text.split("\n")
    line_height = TITLE_FONT_SIZE * 1.2
    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font, stroke_width=TITLE_STROKE)
        widths.append(bbox[2] - bbox[0])
    text_width = max(widths) if widths else 0
    text_height = len(lines) * line_height
    return (x - text_width / 2, y - text_height / 2, x + text_width / 2, y + text_height / 2)


def boxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def place_titles(
    pois: List[POI],
    dynamic_data: DynamicData,
    map_width: int,
    map_height: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    castle_offset_x: float = CASTLE_RENDER_OFFSET_X,
    castle_offset_y: float = CASTLE_RENDER_OFFSET_Y,
) -> Dict[int, TitlePlacement]:
    titles: List[TitlePlacement] = []
    for poi in pois:
        icon_file = icon_for_poi(poi, dynamic_data.map_layout)
        title_info = title_text_for_poi(poi, dynamic_data, icon_file)
        if not title_info:
            continue
        text, priority = title_info
        x, y = scaled_xy(poi.x, poi.y, map_width, map_height, dynamic_data.map_layout, poi.id)
        if poi.id == 159:
            x += castle_offset_x
            y += castle_offset_y
        titles.append(TitlePlacement(poi.id, x, y, text, priority))

    titles.sort(key=lambda t: t.priority)
    placed: List[TitlePlacement] = []

    for title in titles:
        current_x, current_y = title.x, title.y
        iterations = 0
        needs_repositioning = True
        while needs_repositioning and iterations < 10:
            iterations += 1
            needs_repositioning = False
            current_bounds = get_text_bounds(current_x, current_y, title.text, font)
            for idx, placed_title in enumerate(placed):
                placed_bounds = get_text_bounds(placed_title.x, placed_title.y, placed_title.text, font)
                if not boxes_overlap(current_bounds, placed_bounds):
                    continue
                dx = max(0.0, placed_bounds[2] - current_bounds[0] + 5.0, current_bounds[2] - placed_bounds[0] + 5.0)
                dy = max(0.0, placed_bounds[3] - current_bounds[1] + 5.0, current_bounds[3] - placed_bounds[1] + 5.0)
                center_x = (current_x + placed_title.x) / 2.0
                center_y = (current_y + placed_title.y) / 2.0
                move_x = -dx / 2.0 if current_x < center_x else dx / 2.0
                move_y = -dy / 2.0 if current_y < center_y else dy / 2.0
                distance = math.hypot(move_x, move_y)
                if distance > TITLE_MAX_MOVEMENT and distance > 0:
                    scale = TITLE_MAX_MOVEMENT / distance
                    move_x *= scale
                    move_y *= scale
                placed_move_x = -move_x
                placed_move_y = -move_y
                current_x += move_x
                current_y += move_y
                placed[idx] = TitlePlacement(
                    id=placed_title.id,
                    x=placed_title.x + placed_move_x,
                    y=placed_title.y + placed_move_y,
                    text=placed_title.text,
                    priority=placed_title.priority,
                )
                needs_repositioning = True
                break
        placed.append(TitlePlacement(title.id, current_x, current_y, title.text, title.priority))

    return {t.id: t for t in placed}


def try_load_font(font_path: Optional[Path], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: List[Path] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ])
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def resized_icon(icon_path: Path, cache: Dict[str, Image.Image]) -> Image.Image:
    key = str(icon_path)
    if key in cache:
        return cache[key]
    img = Image.open(icon_path).convert("RGBA")
    size_cfg = POI_ICON_SIZES.get(icon_path.name, {})
    if "width" in size_cfg and "height" not in size_cfg:
        w = int(size_cfg["width"])
        h = int(round(img.height / img.width * w))
        img = img.resize((w, h), Image.LANCZOS)
    elif "height" in size_cfg and "width" not in size_cfg:
        h = int(size_cfg["height"])
        w = int(round(img.width / img.height * h))
        img = img.resize((w, h), Image.LANCZOS)
    elif "width" in size_cfg and "height" in size_cfg:
        img = img.resize((int(size_cfg["width"]), int(size_cfg["height"])), Image.LANCZOS)
    cache[key] = img
    return img


def stitch_map_tiles(map_dir: Path, map_layout: str) -> Image.Image:
    tile_dir = {
        "default": map_dir / "default_map_tiles",
        "the_crater_shifted": map_dir / "the_crater_shifted_map_tiles",
        "the_mountaintop_shifted": map_dir / "the_mountaintop_shifted_map_tiles",
        "the_rotten_woods_shifted": map_dir / "the_rotten_woods_shifted_map_tiles",
        "noklateo_shifted": map_dir / "noklateo_shifted_map_tiles",
        "great_hollow_shifted": map_dir / "great_hollow_shifted_map_tiles",
    }[map_layout]

    tiles: List[List[Image.Image]] = []
    for row in range(GRID_SIZE):
        row_tiles = []
        for col in range(GRID_SIZE):
            filename = f"MENU_MapTile_L0_{row:02d}_{col:02d}_webp.webp"
            path = tile_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing tile: {path}")
            row_tiles.append(Image.open(path).convert("RGBA"))
        tiles.append(row_tiles)

    tile_width = tiles[0][0].width
    tile_height = tiles[0][0].height
    canvas = Image.new("RGBA", (tile_width * GRID_SIZE, tile_height * GRID_SIZE), (0, 0, 0, 0))

    for new_row in range(GRID_SIZE):
        for new_col in range(GRID_SIZE):
            old_row = new_col
            old_col = 5 - new_row
            tile = tiles[old_row][old_col]
            canvas.alpha_composite(tile, dest=(new_col * tile_width, new_row * tile_height))

    return canvas


def filter_renderable_pois(
    pois: List[POI],
    dynamic_data: DynamicData,
    hide_field_bosses: bool = False,
    hide_night_bosses: bool = False,
    hide_spawn_location: bool = False,
) -> List[POI]:
    filtered: List[POI] = []
    for poi in pois:
        icon_file = icon_for_poi(poi, dynamic_data.map_layout)
        if hide_field_bosses and icon_file in {"Field_Boss.png", "Field_Boss_Red.png", "Arena_Boss.png"}:
            continue
        if hide_night_bosses and icon_file == "Night_Location.png":
            continue
        if hide_spawn_location and icon_file == "Spawn_Location.png":
            continue
        filtered.append(poi)
    return filtered


def render_layout(
    repo: Path,
    paths: Dict[str, Path],
    layout_number: int,
    show_text: bool,
    font_path: Optional[Path] = None,
    hide_field_bosses: bool = False,
    hide_night_bosses: bool = False,
    hide_spawn_location: bool = False,
    castle_offset_x: float = CASTLE_RENDER_OFFSET_X,
    castle_offset_y: float = CASTLE_RENDER_OFFSET_Y,
) -> Image.Image:
    layout_path = paths["layouts"] / f"layout_{layout_number:03d}.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing layout JSON: {layout_path}")
    layout_data = load_json(layout_path)

    value_icon_map = parse_value_icon_mapping(paths["icon_mapping"].read_text(encoding="utf-8"))
    location_map = parse_location_mapping(paths["location_mapping"].read_text(encoding="utf-8"))
    context_maps = build_context_mappings(paths["location_mapping"].read_text(encoding="utf-8"))
    master_poi = load_master_poi(paths["master_poi"])

    dynamic_data = load_dynamic_data(layout_number, layout_data, master_poi, value_icon_map, location_map, context_maps)
    if DEBUG_CASTLE_ENABLED:
        debug_castle(
            f"layout={layout_number}, shifting_earth={dynamic_data.shifting_earth}, map_layout={dynamic_data.map_layout}, "
            f"special_event={dynamic_data.layout_data.get('Special Event')}"
        )
    coord_layout = load_coordinate_layout(paths["coord_dir"], dynamic_data.map_layout)
    pois = build_all_pois(coord_layout, master_poi, dynamic_data)
    pois = filter_renderable_pois(
        pois,
        dynamic_data,
        hide_field_bosses=hide_field_bosses,
        hide_night_bosses=hide_night_bosses,
        hide_spawn_location=hide_spawn_location,
    )

    base = stitch_map_tiles(paths["map_dir"], dynamic_data.map_layout)
    draw = ImageDraw.Draw(base)
    font = try_load_font(font_path, TITLE_FONT_SIZE)
    title_map = place_titles(
        pois,
        dynamic_data,
        base.width,
        base.height,
        font,
        castle_offset_x=castle_offset_x,
        castle_offset_y=castle_offset_y,
    ) if show_text else {}
    icon_cache: Dict[str, Image.Image] = {}

    for poi in pois:
        sx, sy = scaled_xy(poi.x, poi.y, base.width, base.height, dynamic_data.map_layout, poi.id)
        if poi.id == 159:
            sx += castle_offset_x
            sy += castle_offset_y
            debug_castle(
                f"castle render offset applied: offset_x={castle_offset_x}, offset_y={castle_offset_y}, "
                f"final_sx={sx:.2f}, final_sy={sy:.2f}"
            )

        icon_file = icon_for_poi(poi, dynamic_data.map_layout)
        if poi.id == 159 or icon_file == "Castle.png":
            debug_castle(
                f"render loop candidate: layout={layout_number}, poi_id={poi.id}, poi_type={poi.poi_type}, "
                f"icon_file={icon_file}, sx={sx:.2f}, sy={sy:.2f}, x={poi.x}, y={poi.y}"
            )
        if not icon_file:
            continue
        icon_path = resolve_icon_path(repo, icon_file)
        if icon_path is None:
            print(f"Warning: Missing icon for {icon_file}; skipping this POI.")
            continue
        icon_img = resized_icon(icon_path, icon_cache)
        base.alpha_composite(icon_img, dest=(int(round(sx - icon_img.width / 2)), int(round(sy - icon_img.height / 2))))

        if show_text and poi.id in title_map:
            placement = title_map[poi.id]
            draw.multiline_text(
                (placement.x, placement.y),
                placement.text,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=TITLE_STROKE,
                align="center",
                anchor="mm",
                spacing=4,
            )

    return base


def discover_layouts(layout_dir: Path, selected: Sequence[int], shifting_earth: Optional[str]) -> List[int]:
    if selected:
        layouts = sorted(set(selected))
    else:
        layouts = []
        for path in sorted(layout_dir.glob("layout_*.json")):
            m = re.search(r"(\d+)", path.stem)
            if m:
                layout_num = int(m.group(1))
                if shifting_earth is None:
                    layouts.append(layout_num)
                else:
                    data = load_json(path)
                    if data.get("Shifting Earth") == shifting_earth:
                        layouts.append(layout_num)
    if shifting_earth is not None and selected:
        filtered = []
        for num in layouts:
            data = load_json(layout_dir / f"layout_{num:03d}.json")
            if data.get("Shifting Earth") == shifting_earth:
                filtered.append(num)
        layouts = filtered
    return layouts


def save_image(img: Image.Image, path: Path, ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = ext.lower()
    if ext in {"jpg", "jpeg"}:
        img.convert("RGB").save(path, quality=95)
    elif ext == "webp":
        img.save(path, format="WEBP", quality=95)
    else:
        img.save(path)


def zip_dir(input_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(input_dir.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)


def resolve_icon_path(repo: Path, icon_name: str):
    poi_dir = repo / "public" / "POI_icons"
    direct = poi_dir / icon_name
    if direct.exists():
        return direct
    stem = Path(icon_name).stem
    for alt in [poi_dir / f"{stem}.webp", poi_dir / f"{stem}.png", poi_dir / f"{stem}.jpg", poi_dir / f"{stem}.jpeg"]:
        if alt.exists():
            return alt
    fallback_name = MANUAL_ICON_FALLBACKS.get(icon_name)
    if fallback_name:
        fallback = poi_dir / fallback_name
        if fallback.exists():
            return fallback
    return None


def pil_rgba_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGBA"), dtype=np.uint8), cv2.COLOR_RGBA2BGR)


def preprocess_for_similarity(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 6.0)
    return gray


def preprocess_for_gradients(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
    return mag


def collect_layout_recognition_metadata(
    repo: Path,
    paths: Dict[str, Path],
    layout_number: int,
) -> Dict[str, object]:
    layout_path = paths["layouts"] / f"layout_{layout_number:03d}.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing layout JSON: {layout_path}")
    layout_data = load_json(layout_path)

    value_icon_map = parse_value_icon_mapping(paths["icon_mapping"].read_text(encoding="utf-8"))
    location_map = parse_location_mapping(paths["location_mapping"].read_text(encoding="utf-8"))
    context_maps = build_context_mappings(paths["location_mapping"].read_text(encoding="utf-8"))
    master_poi = load_master_poi(paths["master_poi"])

    dynamic_data = load_dynamic_data(layout_number, layout_data, master_poi, value_icon_map, location_map, context_maps)
    coord_layout = load_coordinate_layout(paths["coord_dir"], dynamic_data.map_layout)
    pois = build_all_pois(coord_layout, master_poi, dynamic_data)
    base = stitch_map_tiles(paths["map_dir"], dynamic_data.map_layout)
    landmarks: List[Dict[str, object]] = []
    for poi in pois:
        icon_file = icon_for_poi(poi, dynamic_data.map_layout)
        landmark_type = LANDMARK_ICON_TO_TYPE.get(icon_file or "")
        if not landmark_type:
            continue
        sx, sy = scaled_xy(poi.x, poi.y, base.width, base.height, dynamic_data.map_layout, poi.id)
        landmarks.append({
            "poi_id": int(poi.id),
            "type": landmark_type,
            "icon": str(icon_file),
            "x": float(sx),
            "y": float(sy),
            "x_norm": float(sx / base.width),
            "y_norm": float(sy / base.height),
        })
    landmarks.sort(key=lambda item: (str(item["type"]), int(item["poi_id"])))
    return {
        "layout_number": int(layout_number),
        "shifting_earth": str(dynamic_data.shifting_earth),
        "map_layout": str(dynamic_data.map_layout),
        "width": int(base.width),
        "height": int(base.height),
        "landmarks": landmarks,
    }


def build_recognition_icon_templates(repo: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    icon_names = ["Church.png", "Fort.png", "Great_Church.png", "Main_Encampment.png", "Ruins.png", "Sorcerer's_Rise.png"]
    icon_cache: Dict[str, Image.Image] = {}
    templates: List[np.ndarray] = []
    gray_templates: List[np.ndarray] = []
    edge_templates: List[np.ndarray] = []
    actual_sizes: List[Tuple[int, int]] = []
    for icon_name in icon_names:
        icon_path = resolve_icon_path(repo, icon_name)
        if icon_path is None:
            raise FileNotFoundError(f"Missing required icon for recognition template: {icon_name}")
        icon_img = resized_icon(icon_path, icon_cache)
        actual_sizes.append((icon_img.width, icon_img.height))
        canvas = Image.new("RGBA", (RECOGNITION_ICON_CANVAS_SIZE, RECOGNITION_ICON_CANVAS_SIZE), (0, 0, 0, 0))
        dx = (RECOGNITION_ICON_CANVAS_SIZE - icon_img.width) // 2
        dy = (RECOGNITION_ICON_CANVAS_SIZE - icon_img.height) // 2
        canvas.alpha_composite(icon_img, dest=(dx, dy))
        rgba = np.array(canvas, dtype=np.uint8)
        templates.append(rgba)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        gray_templates.append(gray)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if float(mag.max()) > 1e-6:
            mag = (mag / float(mag.max()) * 255.0)
        edge_templates.append(mag.astype(np.uint8))
    return (
        icon_names,
        np.stack(templates, axis=0),
        np.array(actual_sizes, dtype=np.int32),
        np.stack(gray_templates, axis=0),
        np.stack(edge_templates, axis=0),
    )


def export_recognition_assets(
    repo: Path,
    paths: Dict[str, Path],
    layouts: Sequence[int],
    package_path: Optional[Path],
    metadata_dir: Optional[Path],
) -> None:
    if package_path is None and metadata_dir is None:
        return

    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    layout_meta: List[Dict[str, object]] = []
    seen_earths: Dict[str, str] = {}
    for layout_number in layouts:
        meta = collect_layout_recognition_metadata(repo, paths, layout_number)
        layout_meta.append(meta)
        seen_earths.setdefault(str(meta["shifting_earth"]), str(meta["map_layout"]))
        if metadata_dir is not None:
            out_path = metadata_dir / f"layout_{int(layout_number):03d}.json"
            out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if package_path is None:
        return

    earth_names = sorted(seen_earths.keys())
    terrain_low: List[np.ndarray] = []
    terrain_grad: List[np.ndarray] = []
    for earth_name in earth_names:
        base = stitch_map_tiles(paths["map_dir"], seen_earths[earth_name])
        base_bgr = pil_rgba_to_bgr(base)
        terrain_low.append(preprocess_for_similarity(base_bgr))
        terrain_grad.append(preprocess_for_gradients(base_bgr))

    icon_names, icon_templates, icon_actual_sizes, icon_templates_gray, icon_templates_edge = build_recognition_icon_templates(repo)
    landmark_types = [LANDMARK_ICON_TO_TYPE[name] for name in icon_names]
    earth_to_idx = {name: idx for idx, name in enumerate(earth_names)}
    type_to_idx = {name: idx for idx, name in enumerate(landmark_types)}

    layout_numbers = np.array([int(meta["layout_number"]) for meta in layout_meta], dtype=np.int32)
    layout_earth_idx = np.array([earth_to_idx[str(meta["shifting_earth"])] for meta in layout_meta], dtype=np.int32)
    offsets = [0]
    landmark_type_idx: List[int] = []
    landmark_x_norm: List[float] = []
    landmark_y_norm: List[float] = []
    landmark_x_px: List[float] = []
    landmark_y_px: List[float] = []
    landmark_source_ids: List[int] = []
    for meta in layout_meta:
        for item in meta["landmarks"]:
            landmark_type_idx.append(type_to_idx[str(item["type"])])
            landmark_x_norm.append(float(item["x_norm"]))
            landmark_y_norm.append(float(item["y_norm"]))
            landmark_x_px.append(float(item["x"]))
            landmark_y_px.append(float(item["y"]))
            landmark_source_ids.append(int(item["poi_id"]))
        offsets.append(len(landmark_type_idx))

    package_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": RECOGNITION_PACKAGE_VERSION,
        "cmp_size": RECOGNITION_CMP_SIZE,
        "icon_canvas_size": RECOGNITION_ICON_CANVAS_SIZE,
        "earth_names": earth_names,
        "landmark_types": landmark_types,
        "icon_names": icon_names,
        "layout_count": len(layout_meta),
        "description": "Prototype geometric recognizer package v3 with position-anchored landmark matching, Sorcerer's Rise included, and gray/edge icon templates.",
    }
    max_earth_len = max((len(name) for name in earth_names), default=1)
    max_type_len = max((len(name) for name in landmark_types), default=1)
    max_icon_len = max((len(name) for name in icon_names), default=1)
    np.savez_compressed(
        package_path,
        metadata_json=np.array(json.dumps(metadata)),
        earth_names=np.array(earth_names, dtype=f"<U{max_earth_len}"),
        terrain_low=np.stack(terrain_low, axis=0),
        terrain_grad=np.stack(terrain_grad, axis=0),
        landmark_types=np.array(landmark_types, dtype=f"<U{max_type_len}"),
        icon_names=np.array(icon_names, dtype=f"<U{max_icon_len}"),
        icon_templates=icon_templates,
        icon_templates_gray=icon_templates_gray,
        icon_templates_edge=icon_templates_edge,
        icon_actual_sizes=icon_actual_sizes,
        layout_numbers=layout_numbers,
        layout_earth_idx=layout_earth_idx,
        layout_offsets=np.array(offsets, dtype=np.int32),
        landmark_type_idx=np.array(landmark_type_idx, dtype=np.int32),
        landmark_x_norm=np.array(landmark_x_norm, dtype=np.float32),
        landmark_y_norm=np.array(landmark_y_norm, dtype=np.float32),
        landmark_x_px=np.array(landmark_x_px, dtype=np.float32),
        landmark_y_px=np.array(landmark_y_px, dtype=np.float32),
        landmark_source_ids=np.array(landmark_source_ids, dtype=np.int32),
    )
    print(f"Saved recognition package {package_path}")




SLOT_CLUSTER_RADIUS_NORM = 0.018
SLOT_ASSIGN_TOLERANCE_FACTOR = 1.55
SLOT_MIN_RADIUS_NORM = 0.018
SLOT_MAX_RADIUS_NORM = 0.040


def _merge_close_slot_clusters(clusters: List[Dict[str, object]], radius: float) -> List[Dict[str, object]]:
    changed = True
    while changed:
        changed = False
        out: List[Dict[str, object]] = []
        used = [False] * len(clusters)
        for i, base in enumerate(clusters):
            if used[i]:
                continue
            used[i] = True
            points = list(base["points"])
            cx = float(base["x_norm"])
            cy = float(base["y_norm"])
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                other = clusters[j]
                dx = float(other["x_norm"]) - cx
                dy = float(other["y_norm"]) - cy
                if math.hypot(dx, dy) <= radius:
                    used[j] = True
                    points.extend(other["points"])
                    arrx = [float(p["x_norm"]) for p in points]
                    arry = [float(p["y_norm"]) for p in points]
                    cx = float(sum(arrx) / len(arrx))
                    cy = float(sum(arry) / len(arry))
                    changed = True
            out.append({"x_norm": cx, "y_norm": cy, "points": points})
        clusters = out
    return clusters


def build_slots_for_earth(layout_meta_for_earth: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
    for meta in layout_meta_for_earth:
        for lm in meta.get("landmarks", []):
            points.append({
                "layout_number": int(meta["layout_number"]),
                "type": str(lm["type"]),
                "x_norm": float(lm["x_norm"]),
                "y_norm": float(lm["y_norm"]),
            })
    points.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))
    clusters: List[Dict[str, object]] = []
    for pt in points:
        best_idx = None
        best_dist = None
        for idx, cluster in enumerate(clusters):
            dx = float(pt["x_norm"]) - float(cluster["x_norm"])
            dy = float(pt["y_norm"]) - float(cluster["y_norm"])
            dist = math.hypot(dx, dy)
            if dist <= SLOT_CLUSTER_RADIUS_NORM and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            clusters.append({"x_norm": float(pt["x_norm"]), "y_norm": float(pt["y_norm"]), "points": [pt]})
        else:
            cluster = clusters[best_idx]
            cluster["points"].append(pt)
            arrx = [float(p["x_norm"]) for p in cluster["points"]]
            arry = [float(p["y_norm"]) for p in cluster["points"]]
            cluster["x_norm"] = float(sum(arrx) / len(arrx))
            cluster["y_norm"] = float(sum(arry) / len(arry))
    clusters = _merge_close_slot_clusters(clusters, SLOT_CLUSTER_RADIUS_NORM * 0.72)

    clusters.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))
    slots: List[Dict[str, object]] = []
    for idx, cluster in enumerate(clusters, start=1):
        cx = float(cluster["x_norm"])
        cy = float(cluster["y_norm"])
        dists = [math.hypot(float(p["x_norm"]) - cx, float(p["y_norm"]) - cy) for p in cluster["points"]]
        max_dist = max(dists) if dists else 0.0
        radius_norm = min(SLOT_MAX_RADIUS_NORM, max(SLOT_MIN_RADIUS_NORM, max_dist * 1.35 + 0.004))
        allowed_types = sorted({str(p["type"]) for p in cluster["points"]})
        slots.append({
            "slot_id": idx,
            "slot_name": f"slot_{idx:02d}",
            "x_norm": cx,
            "y_norm": cy,
            "radius_norm": radius_norm,
            "allowed_types": allowed_types,
        })
    return slots


def assign_landmarks_to_slots(layout_meta: Dict[str, object], slots: Sequence[Dict[str, object]]) -> List[str]:
    slot_labels = ["empty"] * len(slots)
    landmarks = list(layout_meta.get("landmarks", []))
    candidates: List[Tuple[float, int, int]] = []
    for lm_idx, lm in enumerate(landmarks):
        lx = float(lm["x_norm"])
        ly = float(lm["y_norm"])
        for slot_idx, slot in enumerate(slots):
            dx = lx - float(slot["x_norm"])
            dy = ly - float(slot["y_norm"])
            dist = math.hypot(dx, dy)
            tol = float(slot["radius_norm"]) * SLOT_ASSIGN_TOLERANCE_FACTOR
            if dist <= tol:
                candidates.append((dist, lm_idx, slot_idx))
    candidates.sort(key=lambda item: item[0])
    used_landmarks = set()
    used_slots = set()
    for dist, lm_idx, slot_idx in candidates:
        if lm_idx in used_landmarks or slot_idx in used_slots:
            continue
        used_landmarks.add(lm_idx)
        used_slots.add(slot_idx)
        slot_labels[slot_idx] = str(landmarks[lm_idx]["type"])

    # Fallback for any unassigned landmark: attach it to its nearest free slot.
    for lm_idx, lm in enumerate(landmarks):
        if lm_idx in used_landmarks:
            continue
        lx = float(lm["x_norm"])
        ly = float(lm["y_norm"])
        best_slot = None
        best_dist = None
        for slot_idx, slot in enumerate(slots):
            if slot_idx in used_slots:
                continue
            dist = math.hypot(lx - float(slot["x_norm"]), ly - float(slot["y_norm"]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_slot = slot_idx
        if best_slot is not None:
            used_slots.add(best_slot)
            slot_labels[best_slot] = str(lm["type"])
    return slot_labels


def compute_slot_weights(layout_metas: Sequence[Dict[str, object]], slots_by_earth: Dict[str, List[Dict[str, object]]]) -> Dict[str, List[float]]:
    weights: Dict[str, List[float]] = {}
    by_earth: Dict[str, List[Dict[str, object]]] = {}
    for meta in layout_metas:
        by_earth.setdefault(str(meta["shifting_earth"]), []).append(meta)
    for earth_name, slots in slots_by_earth.items():
        metas = by_earth.get(earth_name, [])
        counts_per_slot: List[Dict[str, int]] = [dict() for _ in slots]
        for meta in metas:
            labels = assign_landmarks_to_slots(meta, slots)
            for idx, label in enumerate(labels):
                counts_per_slot[idx][label] = counts_per_slot[idx].get(label, 0) + 1
        slot_weights: List[float] = []
        for counts in counts_per_slot:
            total = max(1, sum(counts.values()))
            probs = [value / total for value in counts.values() if value > 0]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            max_entropy = math.log(max(2, len(counts)))
            norm_entropy = entropy / max_entropy if max_entropy > 1e-12 else 0.0
            slot_weights.append(0.75 + 1.0 * norm_entropy)
        weights[earth_name] = slot_weights
    return weights


def export_recognition_assets(
    repo: Path,
    paths: Dict[str, Path],
    layouts: Sequence[int],
    package_path: Optional[Path],
    metadata_dir: Optional[Path],
) -> None:
    if package_path is None and metadata_dir is None:
        return

    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    layout_meta: List[Dict[str, object]] = []
    seen_earths: Dict[str, str] = {}
    by_earth: Dict[str, List[Dict[str, object]]] = {}
    for layout_number in layouts:
        meta = collect_layout_recognition_metadata(repo, paths, layout_number)
        layout_meta.append(meta)
        seen_earths.setdefault(str(meta["shifting_earth"]), str(meta["map_layout"]))
        by_earth.setdefault(str(meta["shifting_earth"]), []).append(meta)

    slots_by_earth: Dict[str, List[Dict[str, object]]] = {
        earth_name: build_slots_for_earth(metas)
        for earth_name, metas in by_earth.items()
    }
    slot_weights = compute_slot_weights(layout_meta, slots_by_earth)

    layouts_export: List[Dict[str, object]] = []
    for meta in layout_meta:
        earth_name = str(meta["shifting_earth"])
        slots = slots_by_earth[earth_name]
        labels = assign_landmarks_to_slots(meta, slots)
        layout_export = {
            "layout_number": int(meta["layout_number"]),
            "shifting_earth": earth_name,
            "map_layout": str(meta["map_layout"]),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "slot_labels": labels,
            "landmarks": meta["landmarks"],
        }
        layouts_export.append(layout_export)
        if metadata_dir is not None:
            out_path = metadata_dir / f"layout_{int(meta['layout_number']):03d}.json"
            out_path.write_text(json.dumps(layout_export, indent=2), encoding="utf-8")

    if metadata_dir is not None:
        earth_slots_out = {
            earth_name: [
                dict(slot, weight=float(slot_weights[earth_name][idx]))
                for idx, slot in enumerate(slots)
            ]
            for earth_name, slots in slots_by_earth.items()
        }
        (metadata_dir / "slot_catalog_by_earth.json").write_text(json.dumps(earth_slots_out, indent=2), encoding="utf-8")

    if package_path is None:
        return

    earth_names = sorted(seen_earths.keys())
    terrain_low: List[np.ndarray] = []
    terrain_grad: List[np.ndarray] = []
    for earth_name in earth_names:
        base = stitch_map_tiles(paths["map_dir"], seen_earths[earth_name])
        base_bgr = pil_rgba_to_bgr(base)
        terrain_low.append(preprocess_for_similarity(base_bgr))
        terrain_grad.append(preprocess_for_gradients(base_bgr))

    icon_names, icon_templates, icon_actual_sizes, icon_templates_gray, icon_templates_edge = build_recognition_icon_templates(repo)
    landmark_types = [LANDMARK_ICON_TO_TYPE[name] for name in icon_names]

    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_metadata = {
        "version": RECOGNITION_PACKAGE_VERSION,
        "cmp_size": RECOGNITION_CMP_SIZE,
        "icon_canvas_size": RECOGNITION_ICON_CANVAS_SIZE,
        "earth_names": earth_names,
        "landmark_types": landmark_types,
        "icon_names": icon_names,
        "slots_by_earth": {
            earth_name: [
                dict(slot, weight=float(slot_weights[earth_name][idx]))
                for idx, slot in enumerate(slots_by_earth[earth_name])
            ]
            for earth_name in earth_names
        },
        "layouts": layouts_export,
        "description": "Prototype slot-based recognizer package v4: terrain classification plus slot occupancy labels (including real empty slots).",
    }
    max_earth_len = max((len(name) for name in earth_names), default=1)
    max_type_len = max((len(name) for name in landmark_types), default=1)
    max_icon_len = max((len(name) for name in icon_names), default=1)
    np.savez_compressed(
        package_path,
        metadata_json=np.array(json.dumps(package_metadata)),
        earth_names=np.array(earth_names, dtype=f"<U{max_earth_len}"),
        terrain_low=np.stack(terrain_low, axis=0),
        terrain_grad=np.stack(terrain_grad, axis=0),
        landmark_types=np.array(landmark_types, dtype=f"<U{max_type_len}"),
        icon_names=np.array(icon_names, dtype=f"<U{max_icon_len}"),
        icon_templates=icon_templates,
        icon_templates_gray=icon_templates_gray,
        icon_templates_edge=icon_templates_edge,
        icon_actual_sizes=icon_actual_sizes,
    )
    print(f"Saved recognition package {package_path}")

def main() -> int:
    global DEBUG_CASTLE_ENABLED, DEBUG_CASTLE_STREAM
    args = parse_cli()
    DEBUG_CASTLE_ENABLED = bool(args.debug_castle or args.debug_castle_log)
    if args.debug_castle_log is not None:
        args.debug_castle_log.parent.mkdir(parents=True, exist_ok=True)
        DEBUG_CASTLE_STREAM = args.debug_castle_log.open("w", encoding="utf-8")
    try:
        repo_root = args.repo if args.repo is not None else runtime_repo_root()
        paths = validate_repo(repo_root)
        modes = determine_modes(args)
        layouts = discover_layouts(paths["layouts"], args.layout, args.filter_shifting_earth)
        if not layouts:
            raise ValueError("No matching layouts found.")

        single_image = len(layouts) == 1 and len(modes) == 1 and args.out is not None
        if single_image:
            img = render_layout(
                repo_root,
                paths,
                layouts[0],
                modes[0].show_text,
                args.font,
                hide_field_bosses=args.hide_field_bosses,
                hide_night_bosses=args.hide_night_bosses,
                hide_spawn_location=args.hide_spawn_location,
                castle_offset_x=args.castle_offset_x,
                castle_offset_y=args.castle_offset_y,
            )
            save_ext = args.ext
            if args.out.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                save_ext = args.out.suffix.lower().lstrip(".")
            save_image(img, args.out, save_ext)
            print(f"Saved {args.out}")
            export_recognition_assets(repo_root, paths, layouts, args.export_recognition_package, args.export_recognition_dir)
            return 0

        out_dir = args.out_dir
        if args.out is not None and not single_image:
            raise ValueError("--out only works when rendering exactly one image. Use --out-dir for batch rendering.")

        for mode in modes:
            mode_dir = out_dir / mode.name
            mode_dir.mkdir(parents=True, exist_ok=True)
            number = args.start_number
            for layout_num in layouts:
                img = render_layout(
                    repo_root,
                    paths,
                    layout_num,
                    mode.show_text,
                    args.font,
                    hide_field_bosses=args.hide_field_bosses,
                    hide_night_bosses=args.hide_night_bosses,
                    hide_spawn_location=args.hide_spawn_location,
                    castle_offset_x=args.castle_offset_x,
                    castle_offset_y=args.castle_offset_y,
                )
                if args.name_mode == "numbered":
                    file_name = f"{number}.{args.ext}"
                    number += 1
                else:
                    file_name = f"layout_{layout_num:03d}.{args.ext}"
                save_image(img, mode_dir / file_name, args.ext)
                print(f"Saved {mode_dir / file_name}")

            if args.zip:
                zip_path = out_dir / f"{mode.name}.zip"
                zip_dir(mode_dir, zip_path)
                print(f"Saved {zip_path}")

        export_recognition_assets(repo_root, paths, layouts, args.export_recognition_package, args.export_recognition_dir)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if DEBUG_CASTLE_STREAM is not None:
            DEBUG_CASTLE_STREAM.close()
            DEBUG_CASTLE_STREAM = None


if __name__ == "__main__":
    raise SystemExit(main())



