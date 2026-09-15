import csv
import json
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import textwrap

try:
    import cv2
except ImportError:
    cv2 = None

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --------------------------------------------------
# Paths
# --------------------------------------------------
DEFAULT_JSON_PATH = "LotResultSmallBaseAndSpot_grouped_by_patternId_14.json"
json_path_raw = str(os.environ.get("NR_RENDER_JSON_PATH", DEFAULT_JSON_PATH)).strip()
json_path = Path(json_path_raw)
if not json_path.exists():
    json_path = Path(resource_path(json_path_raw))

backgrounds_dir = Path(resource_path("maps"))
icons_dir = Path(resource_path("POI_icons"))
recognition_package_path = Path(
    os.environ.get(
        "NR_RECOGNITION_PACKAGE_PATH",
        r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\nightreign_geom_proto_v4_slots.npz",
    )
)
GREAT_HOLLOW_PATTERNSET_DIR = Path(resource_path("great_hollow_pattern_sets"))

MISSING_ICON_FILES = set()
USED_ICON_FILES = set()

RECOGNITION_PACKAGE_VERSION = 1
RECOGNITION_CMP_SIZE = 512
RECOGNITION_ICON_CANVAS_SIZE = 192
SLOT_CLUSTER_RADIUS_NORM = 0.018
SLOT_ASSIGN_TOLERANCE_FACTOR = 1.55
SLOT_MIN_RADIUS_NORM = 0.018
SLOT_MAX_RADIUS_NORM = 0.040
MAJOR_LOCATION_TYPES = {
    "Fort",
    "Fortified Village",
    "Great Church",
    "Main Encampment",
    "Ruins",
    "Swamp",
}

RECOGNITION_ICON_SPECS = [
    ("Church", "Church.png"),
    ("Fort", "Fort.png"),
    ("Fortified Village", "Fortified_Village.png"),
    ("Great Church", "Great_Church.png"),
    ("Main Encampment", "Camp.png"),
    ("Ruins", "Ruins.png"),
    ("Sorcerer's Rise", "Sorcerer's_Rise.png"),
    ("Swamp", "Swamp.png"),
    ("Township", "Township.png"),
]

RECOGNITION_LANDMARK_TYPE_BY_ICON = {
    "Church": "Church",
    "Fort": "Fort",
    "Fortified Village": "Fortified Village",
    "Great Church": "Great Church",
    "Camp": "Main Encampment",
    "Ruins": "Ruins",
    "Sorcerer's Rise": "Sorcerer's Rise",
    "Sorcerer's Rise Ruined": "Sorcerer's Rise",
    "Swamp": "Swamp",
    "Township": "Township",
}
GREAT_HOLLOW_STATUS_TYPES = {
    "Status_Poison",
    "Status_Bleed",
    "Status_Sleep",
    "Status_Lightning",
    "Status_Rot",
    "Status_Holy",
    "Status_Fire",
    "Status_Magic",
    "Status_Frost",
}
GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE = {
    "Ruins": {"Status_Poison", "Status_Bleed"},
    "Great Church": {"Status_Sleep", "Status_Lightning", "Status_Frost", "Status_Holy"},
    "Fortress": {"Status_Rot", "Status_Holy"},
    "Palace": {"Status_Fire", "Status_Magic"},
}
GREAT_HOLLOW_PATTERNSET_DIR = Path("great_hollow_pattern_sets")
GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD = {
    "greatHollowEastFortMainModifier1": ("Fortress", "Status_Rot"),
    "greatHollowEastFortMainModifier2": ("Fortress", "Status_Holy"),
    "greatHollowWestFortMainModifier1": ("Palace", "Status_Fire"),
    "greatHollowWestFortMainModifier2": ("Palace", "Status_Magic"),
}
GREAT_HOLLOW_SYNTHETIC_POINTS = {
    "Fortress": {"x": 734.399, "z": 267.268, "source_id_suffix": 91},
    "Palace": {"x": 278.752, "z": 805.636, "source_id_suffix": 92},
}
GREAT_HOLLOW_SYNTHETIC_BASE_SCALE = 0.6
GREAT_HOLLOW_SYNTHETIC_STATUS_SCALE = 0.384 * 0.6
GREAT_HOLLOW_SYNTHETIC_STATUS_DX_FACTOR = 0.7
GREAT_HOLLOW_SYNTHETIC_STATUS_DY_FACTOR = 0.66

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
RENDER_ANY_TEXT = True

HIDE_ALL_BOSSES_AND_ALL_TEXT = os.environ.get("NR_HIDE_ALL_BOSSES_AND_ALL_TEXT", "0") == "1"
RENDERER_DEBUG = os.environ.get("NR_RENDERER_DEBUG", "0") == "1"
ENABLE_APPEND_EXTRA_BOSS_NAMES = os.environ.get("NR_RENDER_APPEND_EXTRA_BOSS_NAMES", "1") == "1"

RENDER_MODE_DIRNAME = "no_text" if HIDE_ALL_BOSSES_AND_ALL_TEXT else "with_text"
render_root = Path(
    os.environ.get(
        "NR_RENDER_OUTPUT_ROOT",
        r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\renders_new",
    )
)
output_dir = render_root / RENDER_MODE_DIRNAME
recognition_metadata_dir = output_dir / "recognition_metadata"
recognizer_display_dir = render_root / f"recognizer_{RENDER_MODE_DIRNAME}"
under_output_dir = render_root / f"{RENDER_MODE_DIRNAME}_under"
under_recognizer_display_dir = render_root / f"recognizer_{RENDER_MODE_DIRNAME}_under"

missing_icons_txt = output_dir / "missing_icons.txt"
used_icons_txt = output_dir / "used_icons.txt"

output_dir.mkdir(parents=True, exist_ok=True)
recognizer_display_dir.mkdir(parents=True, exist_ok=True)
under_output_dir.mkdir(parents=True, exist_ok=True)
under_recognizer_display_dir.mkdir(parents=True, exist_ok=True)

GREAT_HOLLOW_UNDER_BACKGROUND_NAME = "Great_Hollow_under"
GREAT_HOLLOW_UNDER_GREAT_CHURCH_X = 470.74
GREAT_HOLLOW_UNDER_GREAT_CHURCH_Z = 110.51
GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_X = 924.0
GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_Z = 394.0
GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_ALLOWED_ICONS = {"Church", "Sorcerer's Rise"}
GREAT_HOLLOW_UNDER_STATUS_BY_VARIATION = {
    "Ancient Heroes of Zamor": "Status_Frost",
    "Inquisitor": "Status_Holy",
}
GREAT_HOLLOW_UNDER_STATUS_BY_MAP_NAME = {
    "Great Church - Ancient Heroes of Zamor": "Status_Frost",
    "Great Church - Inquisitor": "Status_Holy",
}
UNDER_VARIANT_RENDER_ENABLED = os.environ.get("NR_RENDER_GH_UNDER_VARIANTS", "1") == "1"
RENDER_ONLY_UNDER_VARIANTS = os.environ.get("NR_RENDER_ONLY_GH_UNDER_VARIANTS", "0") == "1"
UNDER_ONLY_TARGET_GREAT_CHURCH = os.environ.get("NR_UNDER_ONLY_TARGET_GREAT_CHURCH", "0") == "1"

# Crop only for selected Shifting Earth / rareMap values.
# Key = rareMap name exactly as it appears in the JSON.
# Value = border size in pixels to crop from each side.
CROP_BORDER_BY_RAREMAP = {
    "Great Hollow": 128,
    # "Noklateo": 128,
    # "The Crater": 128,
    # "Mountaintop": 0,   # omit or use 0 for no crop
}

# --------------------------------------------------
# Which patternIds to render
# None means render all
# --------------------------------------------------
# PATTERN_IDS_TO_RENDER = set(range(1000, 1010))
# PATTERN_IDS_TO_RENDER = { 1099, 1100, 1105, 1110, 1115, 1137 }
# PATTERN_IDS_TO_RENDER = { 90,112,160,174,188,191,1139 }
PATTERN_IDS_TO_RENDER = None

# Examples:
# PATTERN_IDS_TO_RENDER = {1005, 1006, 1007, 1008, 1009}
# PATTERN_IDS_TO_RENDER = set(range(1005, 1010))

GREAT_HOLLOW_RESONATING_CRYSTAL_OFF_POINTS = [
    {"x": 127.43, "z": 551.02, "icon": "Resonating_Crystal_Off"},
    {"x": 161.55, "z": 358.17, "icon": "Resonating_Crystal_Off"},
    {"x": 79.38, "z": 303.12, "icon": "Resonating_Crystal_Off"},
    {"x": 248.05, "z": 118.06, "icon": "Resonating_Crystal_Off"},
    {"x": 391.26, "z": 73.57, "icon": "Resonating_Crystal_Off"},
    {"x": 415.84, "z": 379.95, "icon": "Resonating_Crystal_Off"},
    {"x": 493.64, "z": 257.73, "icon": "Resonating_Crystal_Off"},
    {"x": 323.45, "z": 323.90, "icon": "Resonating_Crystal_Off"},
    {"x": 360.88, "z": 538.65, "icon": "Resonating_Crystal_Off"},
    {"x": 305.15, "z": 602.78, "icon": "Resonating_Crystal_Off"},
    {"x": 443.03, "z": 787.01, "icon": "Resonating_Crystal_Off"},
    {"x": 703.39, "z": 135.55, "icon": "Resonating_Crystal_Off"},
    {"x": 711.49, "z": 272.06, "icon": "Resonating_Crystal_Off"},
    {"x": 766.35, "z": 358.98, "icon": "Resonating_Crystal_Off"},
    {"x": 730.68, "z": 548.39, "icon": "Resonating_Crystal_Off"},
    {"x": 548.18, "z": 611.36, "icon": "Resonating_Crystal_Off"},
    {"x": 957.23, "z": 112.36, "icon": "Resonating_Crystal_Off"},
    {"x": 936.45, "z": 341.40, "icon": "Resonating_Crystal_Off"},
    {"x": 829.04, "z": 490.48, "icon": "Resonating_Crystal_Off"},
    {"x": 864.70, "z": 538.59, "icon": "Resonating_Crystal_Off"},
    {"x": 812.49, "z": 792.77, "icon": "Resonating_Crystal_Off"},
    {
        "x": 180.0,
        "z": 1120.0,
        "icon": "Resonating_Crystal_Off",
        "text": "Possible crystal locations",
        "render_scale": 0.85,
        "text_dx": 118,
        "text_dy": 0,
        "render_on_top": True,
    },
]

FIXED_POINTS_BY_RAREMAP = {
   #  "Limveld": [
   #      {"x": 441.0, "z": 213.0, "icon": "Field Boss Red"},
   #      {"x": 500.0, "z": 300.0, "icon": "Church"},
   #  ],
   #  "Crater": [
   #      {"x": 620.0, "z": 410.0, "icon": "Castle"},
   #  ],
   #  "Noklateo": [
   #      {"x": 700.0, "z": 520.0, "icon": "Night Horde"},
   #  ],
   "Great Hollow": [
   #  {"x": 143.604, "z": 392.848, "icon": "Ruins"},
    {"x": 386.488, "z": 926.106, "icon": "Church"},
    # Fixed Church slot (requested):
    # gridXNo=44, gridZNo=38, posX=-81.187, posZ=-12.41, world X=558.813, Z=627.59
    {"x": 558.813, "z": 627.59, "icon": "Church"},
    {"x": 144.499, "z": 618.496, "icon": "Divine Tower", "render_scale": 0.6},
    {"x": 846.955, "z": 219.805, "icon": "Divine Tower", "render_scale": 0.6},
    *GREAT_HOLLOW_RESONATING_CRYSTAL_OFF_POINTS,
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
    # "Sorcerer's Rise",
}

NO_TEXT_ICON_EXCLUSIONS = {
    "Event",
    "Meteor_Strike",
    "Fallingstar_Beast",
    "Night_Horde",
    "Day_1",
    "Walking_Mausoleum",
}
RESONATING_CRYSTAL_OFF_ICON_NAME = "Resonating_Crystal_Off"

# --------------------------------------------------
# Per-position render rules for row items
# Matching is done on the ORIGINAL world-space X/Z values
# before any movement is applied.
#
# Supported keys:
# - rareMap
# - icon_name
# - x
# - z
# - render_icon
# - render_text
# - render_on_top
# - icon_dx
# - icon_dy
# - text_dx
# - text_dy
# - icon_dx_factor
# - icon_dy_factor
# - text_dx_factor
# - text_dy_factor
# - render_layer
# - render_scale
# - overlay_icon_name
# - overlay_render_scale
# - overlay_dx
# - overlay_dy
# - overlay_dx_factor
# - overlay_dy_factor
# - overlay_layer_offset
# - overlay_render_on_top
# - map_name
# - map_name_in
# - variation_name
# - icon_name_override
#
# factor values are multiplied by icon width/height
# --------------------------------------------------
ROW_RENDER_RULES = [
    # Preserve your current Field Boss / Field Boss Red special case
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

    # Preserve the "Basement Boss renders on top" behavior
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
        "rareMap": "Great Hollow",
        "icon_name": "Church",
        "x": 924.0,
        "z": 394.0,
        "render_icon": False,
        "render_text": False,
    },
    {
        "rareMap": "Great Hollow",
        "icon_name": "Sorcerer's Rise",
        "x": 924.0,
        "z": 394.0,
        "render_icon": False,
        "render_text": False,
    },
    {
        "icon_name": "Evergaol",
        "render_scale": 0.8,
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
         "overlay_dx": 0,
         "overlay_dy": 0,
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
         "overlay_dx": 0,
         "overlay_dy": 0,
         "overlay_dx_factor": 0.7,
         "overlay_dy_factor": 0.66,
         "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Great Church",
        "map_name": "Great Church - Nox Warriors",
         "overlay_icon_name": "Status_Lightning",
         "overlay_render_scale": 0.384,
         "overlay_dx": 0,
         "overlay_dy": 0,
         "overlay_dx_factor": 0.7,
         "overlay_dy_factor": 0.66,
         "overlay_layer_offset": 1,
    },
    {
        "icon_name": "Great Church",
        "map_name": "Great Church - Abductor Virgins",
         "overlay_icon_name": "Status_Sleep",
         "overlay_render_scale": 0.384,
         "overlay_dx": 0,
         "overlay_dy": 0,
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
    # Example of icon override: exact smallBaseMapName + exact combination of location/variation
    # {
    #     "map_name": "Field Boss - Ancestor Spirit",
    #     "icon_name": "Field Boss",
    #     "variation_name": "Ancestor Spirit",
    #     "icon_name_override": "Field Boss Red",
    # },

    # Example: smallBaseMapName in a list
    # {
    #     "map_name_in": [
    #         "Field Boss - Ancestor Spirit",
    #         "Field Boss - Bell Bearing Hunter",
    #         "Field Boss - Death Rite Bird",
    #     ],
    #     "icon_name": "Field Boss",
    #     "icon_name_override": "Field Boss Red",
    # },

    # Example: match by combination of smallBaseLocationName + smallBaseVariationName
    # {
    #     "icon_name": "Night Horde",
    #     "variation_name": "Ant Horde",
    #     "icon_name_override": "Ant Horde Special",
    # },

def file_safe_name(name):
    return str(name).strip().replace(" ", "_")

def is_boss_icon_name(icon_name):
    name = str(icon_name).strip().lower()
    if name == "":
        return False

    # Explicit known boss icon names in your renderer
    explicit = {
        "field boss",
        "field boss red",
        "basement boss",
    }

    if name in explicit:
        return True

    # Broad fallback for any other boss-like icon names
    return "boss" in name

def float_matches(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def is_great_hollow_under_great_church_row(rare_map, icon_name, x_world, z_world):
    if str(rare_map).strip() != "Great Hollow":
        return False
    if str(icon_name).strip() != "Great Church":
        return False
    return float_matches(x_world, GREAT_HOLLOW_UNDER_GREAT_CHURCH_X) and float_matches(z_world, GREAT_HOLLOW_UNDER_GREAT_CHURCH_Z)

def is_great_hollow_under_recognition_slot_position(rare_map, x_world, z_world):
    if str(rare_map).strip() != "Great Hollow":
        return False
    return float_matches(x_world, GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_X) and float_matches(z_world, GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_Z)

def is_great_hollow_under_recognition_slot_row(rare_map, icon_name, x_world, z_world):
    if is_great_hollow_under_great_church_row(rare_map, icon_name, x_world, z_world):
        return True
    return is_great_hollow_under_recognition_slot_position(
        rare_map=rare_map,
        x_world=x_world,
        z_world=z_world,
    )


def resolve_great_hollow_under_status_icon(map_name, variation_name):
    map_name = "" if map_name is None else str(map_name).strip()
    variation_name = "" if variation_name is None else str(variation_name).strip()
    status_icon = GREAT_HOLLOW_UNDER_STATUS_BY_MAP_NAME.get(map_name)
    if status_icon:
        return status_icon
    return GREAT_HOLLOW_UNDER_STATUS_BY_VARIATION.get(variation_name, "")


def has_great_hollow_under_variant(entry):
    if str(entry.get("rareMap", "")).strip() != "Great Hollow":
        return False
    for row in entry.get("rows", []):
        if is_great_hollow_under_great_church_row(
            rare_map=entry.get("rareMap", ""),
            icon_name=row.get("smallBaseLocationName", ""),
            x_world=row.get("X", ""),
            z_world=row.get("Z", ""),
        ):
            variation = str(row.get("smallBaseVariationName", "")).strip()
            map_name = str(row.get("smallBaseMapName", "")).strip()
            if resolve_great_hollow_under_status_icon(map_name, variation) != "":
                return True
    return False


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

    if HIDE_ALL_BOSSES_AND_ALL_TEXT and normalize_row_icon_name(icon_name) in NO_TEXT_ICON_EXCLUSIONS:
        config["render_icon"] = False
        config["render_text"] = False
        return config

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

def wrap_text_by_words(text, max_chars=16):
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

def world_to_image(x, z):
    """
    Convert world-space (X, Z) to image-space (px, py).
    X grows left->right.
    Z grows bottom->top, so Y is inverted for the image.
    """
    px = (float(x) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = IMAGE_H - ((float(z) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H)

    px = max(0, min(IMAGE_W - 1, int(round(px))))
    py = max(0, min(IMAGE_H - 1, int(round(py))))
    return px, py

def load_font(size):
    """
    Tries a few common fonts, then falls back to PIL default.
    """
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

def draw_label(draw, center_x, center_y, text, font, fill="white", stroke_fill="black", stroke_width=1, max_chars=16, align="center"):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align=align,
        spacing=4,
        stroke_width=stroke_width,
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if align == "center":
        x = center_x - text_w / 2
        y = center_y - text_h / 2
    elif align == "left":
        x = center_x
        y = center_y - text_h / 2
    else:
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
        align=align,
    )

def get_label_bbox(draw, center_x, center_y, text, font, stroke_width=1, max_chars=16, align="center"):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align=align,
        spacing=4,
        stroke_width=stroke_width,
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if align == "center":
        x = center_x - text_w / 2
        y = center_y - text_h / 2
    elif align == "left":
        x = center_x
        y = center_y - text_h / 2
    else:
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
    align="center",
):
    # Try the original position first
    wrapped_text, bbox = get_label_bbox(
        draw, center_x, center_y, text, font,
        stroke_width=stroke_width,
        max_chars=max_chars,
        align=align
    )

    if not any(boxes_overlap(bbox, old_box, padding=padding) for old_box in placed_boxes):
        return center_x, center_y, bbox

    # Then search outward in rings
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
                max_chars=max_chars,
                align=align
            )

            if not any(boxes_overlap(test_bbox, old_box, padding=padding) for old_box in placed_boxes):
                return test_x, test_y, test_bbox

    # Fallback: keep original position if no free spot found
    return center_x, center_y, bbox

def paste_centered(base_image, overlay, center_x, center_y):
    x = int(center_x - overlay.width / 2)
    y = int(center_y - overlay.height / 2)
    base_image.paste(overlay, (x, y), overlay)

def append_landmark(
    landmarks,
    *,
    landmark_type,
    x,
    y,
    width,
    height,
    icon="",
    text="",
    source="",
    source_id=None,
):
    landmarks.append({
        "type": str(landmark_type),
        "icon": str(icon),
        "x": float(x),
        "y": float(y),
        "x_norm": float(x) / float(width),
        "y_norm": float(y) / float(height),
        "text": str(text),
        "source": str(source),
        "source_id": None if source_id is None else int(source_id),
    })

def append_great_hollow_under_recognition_slot_landmarks(landmarks, rare_map, width, height):
    if str(rare_map).strip() != "Great Hollow":
        return
    px, py = world_to_image(
        GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_X,
        GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_Z,
    )
    append_landmark(
        landmarks,
        landmark_type="under_slot_gh_43_38",
        icon="",
        x=px,
        y=py,
        width=width,
        height=height,
        text="",
        source="under_slot_anchor",
        source_id=43_038,
    )

def build_annotation(pattern_id, rare_map, width, height, landmarks, crop_border):
    return {
        "patternId": int(pattern_id),
        "rareMap": str(rare_map),
        "width": int(width),
        "height": int(height),
        "cropBorder": int(crop_border),
        "landmarks": landmarks,
    }

def apply_crop_to_landmarks(landmarks, crop_px, cropped_width, cropped_height):
    if crop_px <= 0:
        return

    for landmark in landmarks:
        landmark["x"] = float(landmark["x"]) - float(crop_px)
        landmark["y"] = float(landmark["y"]) - float(crop_px)
        landmark["x_norm"] = float(landmark["x"]) / float(cropped_width)
        landmark["y_norm"] = float(landmark["y"]) / float(cropped_height)

def pil_rgba_to_bgr(img):
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    if cv2 is not None:
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return rgba[:, :, :3][:, :, ::-1]

def preprocess_for_similarity(img_bgr):
    if cv2 is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (0, 0), 6.0)
        return gray

    rgb = Image.fromarray(img_bgr[:, :, ::-1], mode="RGB")
    gray_img = rgb.convert("L").resize((RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), Image.Resampling.LANCZOS)
    gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=6.0))
    return np.array(gray_img, dtype=np.uint8)

def preprocess_for_gradients(img_bgr):
    if cv2 is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (0, 0), 2.2)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
        return mag

    rgb = Image.fromarray(img_bgr[:, :, ::-1], mode="RGB")
    gray_img = rgb.convert("L").resize((RECOGNITION_CMP_SIZE, RECOGNITION_CMP_SIZE), Image.Resampling.LANCZOS)
    gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=2.2))
    gray = np.array(gray_img, dtype=np.float32)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy).astype(np.float32)
    return mag

def load_background_image(rare_map):
    background_path = backgrounds_dir / f"{file_safe_name(rare_map)}.png"
    if not background_path.exists():
        raise FileNotFoundError(f"Missing background {background_path}")
    bg = Image.open(background_path).convert("RGBA")
    if bg.size != (IMAGE_W, IMAGE_H):
        bg = bg.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
    crop_px = int(CROP_BORDER_BY_RAREMAP.get(str(rare_map).strip(), 0))
    if crop_px > 0:
        bg = bg.crop((crop_px, crop_px, bg.width - crop_px, bg.height - crop_px))
    return bg

def build_recognition_icon_templates():
    templates = []
    gray_templates = []
    edge_templates = []
    actual_sizes = []
    icon_names = []
    landmark_types = []

    for landmark_type, icon_file in RECOGNITION_ICON_SPECS:
        icon_path = icons_dir / icon_file
        if not icon_path.exists():
            raise FileNotFoundError(f"Missing required recognition icon: {icon_path}")
        icon_img = Image.open(icon_path).convert("RGBA")
        target_size = int(BASE_ICON_SIZE * ICON_SCALE)
        orig_w, orig_h = icon_img.size
        scale = target_size / max(orig_w, orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        actual_sizes.append((icon_img.width, icon_img.height))

        canvas = Image.new("RGBA", (RECOGNITION_ICON_CANVAS_SIZE, RECOGNITION_ICON_CANVAS_SIZE), (0, 0, 0, 0))
        dx = (RECOGNITION_ICON_CANVAS_SIZE - icon_img.width) // 2
        dy = (RECOGNITION_ICON_CANVAS_SIZE - icon_img.height) // 2
        canvas.alpha_composite(icon_img, dest=(dx, dy))

        rgba = np.array(canvas, dtype=np.uint8)
        if cv2 is not None:
            gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
            edge = cv2.Canny(gray, 24, 96)
        else:
            gray = np.array(Image.fromarray(rgba, mode="RGBA").convert("L"), dtype=np.uint8)
            edge_img = Image.fromarray(gray, mode="L").filter(ImageFilter.FIND_EDGES)
            edge = np.array(edge_img, dtype=np.uint8)

        icon_names.append(icon_file)
        landmark_types.append(landmark_type)
        templates.append(rgba)
        gray_templates.append(gray)
        edge_templates.append(edge)

    return (
        icon_names,
        landmark_types,
        np.stack(templates, axis=0),
        np.array(actual_sizes, dtype=np.int32),
        np.stack(gray_templates, axis=0),
        np.stack(edge_templates, axis=0),
    )

def collect_pattern_recognition_metadata(pattern_id, rare_map, annotation):
    landmarks = []
    great_hollow_statuses = []
    for item in annotation.get("landmarks", []):
        landmark_type = RECOGNITION_LANDMARK_TYPE_BY_ICON.get(str(item.get("type", "")).strip())
        if not landmark_type:
            continue
        landmarks.append({
            "type": landmark_type,
            "icon": str(item.get("icon", "")),
            "x": float(item["x"]),
            "y": float(item["y"]),
            "x_norm": float(item["x_norm"]),
            "y_norm": float(item["y_norm"]),
        })
    landmarks.sort(key=lambda entry: (str(entry["type"]), float(entry["y_norm"]), float(entry["x_norm"])))
    if str(rare_map) == "Great Hollow":
        by_source_id = {}
        for item in annotation.get("landmarks", []):
            source_id = item.get("source_id", None)
            by_source_id.setdefault(source_id, []).append(item)
        for source_id, items in by_source_id.items():
            overlay_item = None
            base_item = None
            for item in items:
                item_type = str(item.get("type", "")).strip()
                if item_type in GREAT_HOLLOW_STATUS_TYPES:
                    overlay_item = item
                if item_type in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
                    base_item = item
            if overlay_item is None or base_item is None:
                continue
            base_type = str(base_item.get("type", "")).strip()
            status_type = str(overlay_item.get("type", "")).strip()
            if status_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(str(base_type), set()):
                continue
            great_hollow_statuses.append({
                "anchor_type": str(base_type),
                "status_type": status_type,
                "x_norm": float(overlay_item["x_norm"]),
                "y_norm": float(overlay_item["y_norm"]),
                "source_id": None if source_id is None else int(source_id),
            })
    great_hollow_statuses.sort(key=lambda entry: (str(entry["anchor_type"]), str(entry["status_type"])))
    out = {
        "layout_number": int(pattern_id),
        "shifting_earth": str(rare_map),
        "map_layout": str(rare_map),
        "width": int(annotation["width"]),
        "height": int(annotation["height"]),
        "landmarks": landmarks,
    }
    if great_hollow_statuses:
        out["great_hollow_statuses"] = great_hollow_statuses
    return out

def _merge_close_slot_clusters(clusters, radius):
    changed = True
    while changed:
        changed = False
        out = []
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
                    cx = float(sum(float(p["x_norm"]) for p in points) / len(points))
                    cy = float(sum(float(p["y_norm"]) for p in points) / len(points))
                    changed = True
            out.append({"x_norm": cx, "y_norm": cy, "points": points})
        clusters = out
    return clusters

def _apply_slot_allowed_type_constraints(allowed_types):
    allowed = [str(name) for name in allowed_types]
    has_major = any(name in MAJOR_LOCATION_TYPES for name in allowed)
    if has_major:
        # Township is only valid for Minor Location slots.
        allowed = [name for name in allowed if name != "Township"]
    return sorted(set(allowed))

def build_slots_for_earth(layout_meta_for_earth):
    points = []
    for meta in layout_meta_for_earth:
        for lm in meta.get("landmarks", []):
            points.append({
                "layout_number": int(meta["layout_number"]),
                "type": str(lm["type"]),
                "x_norm": float(lm["x_norm"]),
                "y_norm": float(lm["y_norm"]),
            })

    points.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))
    clusters = []
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
            cluster["x_norm"] = float(sum(float(p["x_norm"]) for p in cluster["points"]) / len(cluster["points"]))
            cluster["y_norm"] = float(sum(float(p["y_norm"]) for p in cluster["points"]) / len(cluster["points"]))

    clusters = _merge_close_slot_clusters(clusters, SLOT_CLUSTER_RADIUS_NORM * 0.72)
    clusters.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))

    slots = []
    for idx, cluster in enumerate(clusters, start=1):
        cx = float(cluster["x_norm"])
        cy = float(cluster["y_norm"])
        dists = [math.hypot(float(p["x_norm"]) - cx, float(p["y_norm"]) - cy) for p in cluster["points"]]
        max_dist = max(dists) if dists else 0.0
        radius_norm = min(SLOT_MAX_RADIUS_NORM, max(SLOT_MIN_RADIUS_NORM, max_dist * 1.35 + 0.004))
        allowed_types = _apply_slot_allowed_type_constraints(
            {str(p["type"]) for p in cluster["points"]}
        )
        allowed_labels = ["empty"] + [type_name for type_name in allowed_types if type_name != "empty"]
        slots.append({
            "slot_id": idx,
            "slot_name": f"slot_{idx:02d}",
            "x_norm": cx,
            "y_norm": cy,
            "radius_norm": radius_norm,
            "allowed_types": allowed_types,
            "allowed_labels": allowed_labels,
        })
    return slots

def assign_landmarks_to_slots(layout_meta, slots):
    slot_labels = ["empty"] * len(slots)
    landmarks = list(layout_meta.get("landmarks", []))
    candidates = []
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
    for _, lm_idx, slot_idx in candidates:
        if lm_idx in used_landmarks or slot_idx in used_slots:
            continue
        used_landmarks.add(lm_idx)
        used_slots.add(slot_idx)
        slot_labels[slot_idx] = str(landmarks[lm_idx]["type"])

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

def compute_slot_weights(layout_metas, slots_by_earth):
    weights = {}
    by_earth = {}
    for meta in layout_metas:
        by_earth.setdefault(str(meta["shifting_earth"]), []).append(meta)

    for earth_name, slots in slots_by_earth.items():
        metas = by_earth.get(earth_name, [])
        counts_per_slot = [dict() for _ in slots]
        for meta in metas:
            labels = assign_landmarks_to_slots(meta, slots)
            for idx, label in enumerate(labels):
                counts_per_slot[idx][label] = counts_per_slot[idx].get(label, 0) + 1

        slot_weights = []
        for counts in counts_per_slot:
            total = max(1, sum(counts.values()))
            probs = [value / total for value in counts.values() if value > 0]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            max_entropy = math.log(max(2, len(counts)))
            norm_entropy = entropy / max_entropy if max_entropy > 1e-12 else 0.0
            slot_weights.append(0.75 + 1.0 * norm_entropy)
        weights[earth_name] = slot_weights
    return weights

def export_recognition_assets(layout_meta):
    if not layout_meta:
        return

    recognition_metadata_dir.mkdir(parents=True, exist_ok=True)
    by_earth = {}
    for meta in layout_meta:
        by_earth.setdefault(str(meta["shifting_earth"]), []).append(meta)

    slots_by_earth = {
        earth_name: build_slots_for_earth(metas)
        for earth_name, metas in by_earth.items()
    }
    slot_weights = compute_slot_weights(layout_meta, slots_by_earth)

    layouts_export = []
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
        if meta.get("great_hollow_statuses"):
            layout_export["great_hollow_statuses"] = list(meta["great_hollow_statuses"])
        layouts_export.append(layout_export)
        out_path = recognition_metadata_dir / f"layout_{int(meta['layout_number']):04d}.json"
        out_path.write_text(json.dumps(layout_export, indent=2), encoding="utf-8")

    earth_slots_out = {
        earth_name: [
            dict(slot, weight=float(slot_weights[earth_name][idx]))
            for idx, slot in enumerate(slots)
        ]
        for earth_name, slots in slots_by_earth.items()
    }
    (recognition_metadata_dir / "slot_catalog_by_earth.json").write_text(json.dumps(earth_slots_out, indent=2), encoding="utf-8")

    earth_names = sorted(by_earth.keys())
    terrain_low = []
    terrain_grad = []
    for earth_name in earth_names:
        base = load_background_image(earth_name)
        base_bgr = pil_rgba_to_bgr(base)
        terrain_low.append(preprocess_for_similarity(base_bgr))
        terrain_grad.append(preprocess_for_gradients(base_bgr))

    icon_names, landmark_types, icon_templates, icon_actual_sizes, icon_templates_gray, icon_templates_edge = build_recognition_icon_templates()
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
        "description": "Slot-based recognition package generated from render_bosses_on_backgrounds.py.",
    }
    max_earth_len = max((len(name) for name in earth_names), default=1)
    max_type_len = max((len(name) for name in landmark_types), default=1)
    max_icon_len = max((len(name) for name in icon_names), default=1)
    recognition_package_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        recognition_package_path,
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
    print(f"Saved recognition package: {recognition_package_path}")

def render_boss(
    icon_queue,
    icon_img,
    boss_name,
    x_world,
    z_world,
    label_queue,
    label_dx=0,
    label_dy=0,
    render_layer=100,
):
    if HIDE_ALL_BOSSES_AND_ALL_TEXT:
        return

    if boss_name is None:
        return
    boss_name = str(boss_name).strip()
    if boss_name == "":
        return

    if x_world in ("", None) or z_world in ("", None):
        return

    px, py = world_to_image(x_world, z_world)

    if icon_queue is not None:
        icon_queue.append((render_layer, icon_img, px, py, "Night Location", boss_name, "boss", None))
    label_queue.append((px + label_dx, py + label_dy, boss_name, boss_font, "boss", None))


def format_night_boss_text(night_index: int, boss_name: str, extra_boss_name: str) -> str:
    lines = [f"Night {int(night_index)}"]
    boss_name_txt = str(boss_name or "").strip()
    if boss_name_txt:
        lines.append(boss_name_txt)
    if ENABLE_APPEND_EXTRA_BOSS_NAMES:
        extra_boss_txt = str(extra_boss_name or "").strip()
        if extra_boss_txt:
            lines.append("extraboss: ")
            lines.append(extra_boss_txt)
    return "\n".join(lines).strip()

# --------------------------------------------------
# NEW: icon cache and row rendering
# --------------------------------------------------
ICON_CACHE = {}

def normalize_row_icon_name(icon_name):
    icon_name = str(icon_name).strip()
    if icon_name == "":
        return ""

    lowered = icon_name.lower()
    if lowered == "field boss red":
        return "Gold Merchant"
    if lowered == "mission objective":
        return "Church"

    return icon_name

def get_row_icon(icon_name, render_scale=1.0):
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
        print(f"Missing row icon: {icon_path.name}")
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

def render_fixed_points(icon_queue, label_queue, rare_map, include_resonating_crystal_points=True):
    fixed_points = FIXED_POINTS_BY_RAREMAP.get(str(rare_map).strip(), [])

    for point_index, point in enumerate(fixed_points):
        icon_name = str(point.get("icon", "")).strip()
        text_value = str(point.get("text", "") or "").strip()
        x_world = point.get("x", None)
        z_world = point.get("z", None)
        render_scale = point.get("render_scale", None)

        if icon_name == "" or x_world is None or z_world is None:
            continue
        if (
            str(icon_name).strip() == RESONATING_CRYSTAL_OFF_ICON_NAME
            and not bool(include_resonating_crystal_points)
        ):
            continue

        if HIDE_ALL_BOSSES_AND_ALL_TEXT and is_boss_icon_name(icon_name):
            continue
        if HIDE_ALL_BOSSES_AND_ALL_TEXT and normalize_row_icon_name(icon_name) in NO_TEXT_ICON_EXCLUSIONS:
            continue

        config = get_row_render_config(
            rare_map=rare_map,
            icon_name=icon_name,
            variation_name="",
            map_name="",
            x_world=x_world,
            z_world=z_world,
        )
        if render_scale is not None:
            config["render_scale"] = render_scale
        for key in (
            "render_icon",
            "render_text",
            "render_on_top",
            "render_layer",
            "icon_dx",
            "icon_dy",
            "text_dx",
            "text_dy",
            "icon_dx_factor",
            "icon_dy_factor",
            "text_dx_factor",
            "text_dy_factor",
        ):
            if key in point:
                config[key] = point.get(key)

        if not config["render_icon"] and not config["render_text"] and text_value == "":
            continue

        px, py = world_to_image(x_world, z_world)

        effective_icon_name = str(config["icon_name_override"]).strip() or icon_name

        icon_img = None
        if config["render_icon"]:
            if HIDE_ALL_BOSSES_AND_ALL_TEXT and is_boss_icon_name(effective_icon_name):
                icon_img = None
            else:
                icon_img = get_row_icon(effective_icon_name, config["render_scale"])
                if icon_img is None and SKIP_TEXT_WHEN_ICON_MISSING:
                    continue

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

            icon_queue.append((render_layer, icon_img, icon_px, icon_py, effective_icon_name, text_value, "fixed_point", point_index))

        if (not HIDE_ALL_BOSSES_AND_ALL_TEXT) and config["render_text"] and text_value != "":
            label_queue.append((text_px, text_py, text_value, boss_font, "fixed_point", point_index))

def render_row_item(icon_queue, row, label_queue, rare_map, render_under_variant=False):
    icon_name = row.get("smallBaseLocationName", "")
    label_text = row.get("smallBaseVariationName", "")
    x_world = row.get("X", "")
    z_world = row.get("Z", "")

    map_name = row.get("smallBaseMapName", "")
    map_name = "" if map_name is None else str(map_name).strip()

    icon_name = "" if icon_name is None else str(icon_name).strip()
    label_text = "" if label_text is None else str(label_text).strip()

    if icon_name == "":
        return
    if x_world in ("", None) or z_world in ("", None):
        return
    if (
        render_under_variant
        and is_great_hollow_under_recognition_slot_position(
            rare_map=rare_map,
            x_world=x_world,
            z_world=z_world,
        )
        and icon_name not in GREAT_HOLLOW_UNDER_RECOGNITION_SLOT_ALLOWED_ICONS
    ):
        # Enforce this under-slot as Church / Sorcerer's Rise / empty only.
        # Non-allowed rows are treated as empty.
        return

    config = get_row_render_config(
        rare_map=rare_map,
        icon_name=icon_name,
        variation_name=label_text,
        map_name=map_name,
        x_world=x_world,
        z_world=z_world,
    )

    if render_under_variant and is_great_hollow_under_great_church_row(
        rare_map=rare_map,
        icon_name=icon_name,
        x_world=x_world,
        z_world=z_world,
    ):
        config["render_icon"] = True
        config["render_text"] = True
        status_icon = resolve_great_hollow_under_status_icon(map_name, label_text)
        if status_icon != "":
            config["overlay_icon_name"] = status_icon
            config["overlay_render_scale"] = 0.384
            config["overlay_dx"] = 0
            config["overlay_dy"] = 0
            config["overlay_dx_factor"] = 0.7
            config["overlay_dy_factor"] = 0.66
            config["overlay_layer_offset"] = 1
    if render_under_variant and is_great_hollow_under_recognition_slot_position(
        rare_map=rare_map,
        x_world=x_world,
        z_world=z_world,
    ):
        # In under renders, explicitly show this recognition slot when present.
        config["render_icon"] = True
        config["render_text"] = True

    if not config["render_icon"] and not config["render_text"] and str(config["overlay_icon_name"]).strip() == "":
        return

    px, py = world_to_image(x_world, z_world)
    text_to_draw = label_text if label_text != "" else icon_name

    effective_icon_name = str(config["icon_name_override"]).strip() or icon_name

    icon_img = None
    if config["render_icon"]:
        if HIDE_ALL_BOSSES_AND_ALL_TEXT and is_boss_icon_name(effective_icon_name):
            icon_img = None
        else:
            icon_img = get_row_icon(effective_icon_name, config["render_scale"])
            if icon_img is None and SKIP_TEXT_WHEN_ICON_MISSING:
                return

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

        icon_queue.append((base_render_layer, icon_img, icon_px, icon_py, effective_icon_name, text_to_draw, "row_item", row.get("ID")))

    overlay_icon_name = str(config["overlay_icon_name"]).strip()
    if overlay_icon_name != "":
        if not (HIDE_ALL_BOSSES_AND_ALL_TEXT and is_boss_icon_name(overlay_icon_name)):
            overlay_img = get_row_icon(overlay_icon_name, config["overlay_render_scale"])
            if overlay_img is not None:
                overlay_px = icon_px + config["overlay_dx"]
                overlay_py = icon_py + config["overlay_dy"]

                overlay_px += int(round(overlay_img.width * config["overlay_dx_factor"]))
                overlay_py += int(round(overlay_img.height * config["overlay_dy_factor"]))

                overlay_layer = base_render_layer + int(config["overlay_layer_offset"])
                if config["overlay_render_on_top"]:
                    overlay_layer = max(overlay_layer, 1001)

                icon_queue.append((overlay_layer, overlay_img, overlay_px, overlay_py, overlay_icon_name, "", "overlay", row.get("ID")))

    if (not HIDE_ALL_BOSSES_AND_ALL_TEXT) and config["render_text"] and text_to_draw != "":
        label_queue.append((text_px, text_py, text_to_draw, boss_font, "row_item", row.get("ID")))


def load_great_hollow_patternset_statuses(pattern_id):
    pattern_path = GREAT_HOLLOW_PATTERNSET_DIR / f"pattern_{int(pattern_id):04d}_mappatternset_bundle.json"
    if RENDERER_DEBUG:
        print(f"DEBUG: Loading pattern {pattern_id}, path: {pattern_path}")
    if not pattern_path.exists():
        if RENDERER_DEBUG:
            print("DEBUG: Pattern file does not exist")
        return {}
    try:
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
        if RENDERER_DEBUG:
            print(f"DEBUG: Payload loaded, keys: {list(payload.keys())}")
    except Exception as e:
        if RENDERER_DEBUG:
            print(f"DEBUG: Failed to load JSON: {e}")
        return {}

    statuses = {}
    fields = payload.get("activeMapPatternSetFields", [])
    if RENDERER_DEBUG:
        print(f"DEBUG: activeMapPatternSetFields: {[item.get('field') for item in fields]}")
    for item in fields:
        field_name = str(item.get("field", "")).strip()
        if RENDERER_DEBUG:
            print(f"DEBUG: Checking field: '{field_name}'")
        mapping = GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD.get(field_name)
        if mapping is None:
            if RENDERER_DEBUG:
                print(f"DEBUG: No mapping for field '{field_name}'")
            continue
        anchor_type, status_type = mapping
        statuses[anchor_type] = status_type
        if RENDERER_DEBUG:
            print(f"DEBUG: Added status {anchor_type}: {status_type}")
    if RENDERER_DEBUG:
        print(f"DEBUG: Final statuses: {statuses}")
    return statuses


def render_great_hollow_hidden_features(icon_queue, label_queue, pattern_id, rare_map):
    if str(rare_map).strip() != "Great Hollow":
        return

    if RENDERER_DEBUG:
        print(f"DEBUG: Rendering GH hidden features for pattern {pattern_id}")

    active_statuses = load_great_hollow_patternset_statuses(pattern_id)
    if RENDERER_DEBUG:
        print(f"DEBUG: Active statuses: {active_statuses}")
    if not active_statuses:
        return

    for anchor_type, status_type in active_statuses.items():
        point = GREAT_HOLLOW_SYNTHETIC_POINTS.get(anchor_type)
        if point is None:
            continue

        px, py = world_to_image(point["x"], point["z"])
        base_img = get_row_icon(anchor_type, GREAT_HOLLOW_SYNTHETIC_BASE_SCALE)
        if base_img is None:
            continue

        source_id = int(pattern_id) * 100 + int(point["source_id_suffix"])
        base_px = px
        base_py = py
        icon_queue.append((0, base_img, base_px, base_py, anchor_type, "", "gh_pattern_set", source_id))

        overlay_img = get_row_icon(status_type, GREAT_HOLLOW_SYNTHETIC_STATUS_SCALE)
        if overlay_img is None:
            continue
        overlay_px = base_px + int(round(overlay_img.width * GREAT_HOLLOW_SYNTHETIC_STATUS_DX_FACTOR))
        overlay_py = base_py + int(round(overlay_img.height * GREAT_HOLLOW_SYNTHETIC_STATUS_DY_FACTOR))
        icon_queue.append((1, overlay_img, overlay_px, overlay_py, status_type, "", "overlay", source_id))


def should_render_pattern(pattern_id):
    if PATTERN_IDS_TO_RENDER is None:
        return True

    try:
        pattern_id = int(pattern_id)
    except (TypeError, ValueError):
        return False

    return pattern_id in PATTERN_IDS_TO_RENDER

# --------------------------------------------------
# Load input JSON
# --------------------------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --------------------------------------------------
# Load shared icon and font
# --------------------------------------------------
boss_icon_path = icons_dir / "Night_Location.png"
if not boss_icon_path.exists():
    raise FileNotFoundError(f"Missing boss icon: {boss_icon_path}")

USED_ICON_FILES.add(boss_icon_path.name)

boss_icon = Image.open(boss_icon_path).convert("RGBA")

target_size = int(BASE_ICON_SIZE * ICON_SCALE)

orig_w, orig_h = boss_icon.size
scale = target_size / max(orig_w, orig_h)

new_w = max(1, int(round(orig_w * scale)))
new_h = max(1, int(round(orig_h * scale)))

boss_icon = boss_icon.resize((new_w, new_h), Image.Resampling.LANCZOS)

title_font = load_font(28)
boss_font = load_font(22)

# --------------------------------------------------
# Render one image per patternId
# --------------------------------------------------
def get_entry_by_pattern_id(pattern_id):
    try:
        pattern_id = int(pattern_id)
    except (TypeError, ValueError):
        raise KeyError(f"Invalid pattern_id: {pattern_id}")

    for entry in data:
        try:
            if int(entry.get("patternId", -1)) == pattern_id:
                return entry
        except (TypeError, ValueError):
            continue

    raise KeyError(f"No entry found for patternId {pattern_id}")


# TargetBoss enum: https://soulsmods.github.io/emedf/nr-emedf.html#TargetBoss
FINAL_BOSS_NAMES = {
    0: "Gladius", 1: "Adel", 2: "Gnoster", 3: "Maris", 4: "Libra",
    5: "Fulghor", 6: "Caligo", 7: "Heolstor",
    8: "Harmonia (Balancers)", 9: "Straghess (Dreglord)",
}


@lru_cache(maxsize=1)
def load_final_boss_names_by_pattern():
    """Read the explicit layout-to-Nightlord mapping from the game table."""
    names = {}
    table_path = Path(resource_path("LotResultMapPatternFlag.csv"))
    if not table_path.exists():
        return names
    with table_path.open(encoding="utf-8-sig", newline="") as table:
        for row in csv.DictReader(table):
            pattern_id = int(row["patternId"])
            name = FINAL_BOSS_NAMES.get(int(row["targetBoss"]))
            if pattern_id in names and names[pattern_id] != name:
                raise ValueError(f"Conflicting final bosses for pattern {pattern_id}")
            names[pattern_id] = name
    return names


def build_pattern_image_and_annotation(
    entry,
    background_name_override=None,
    render_under_variant=False,
    render_background=True,
    render_icons=True,
    render_labels=True,
    render_night_location_icons=False,
    render_resonating_crystal_icons=False,
    include_resonating_crystal_points=True,
):
    pattern_id = entry.get("patternId", "")
    rare_map = str(entry.get("rareMap", "")).strip()
    under_target_only = bool(render_under_variant)

    if not should_render_pattern(pattern_id):
        return None
    if rare_map == "":
        print(f"Skipping pattern {pattern_id}: rareMap is empty")
        return None

    if render_background:
        background_name = str(background_name_override).strip() if background_name_override else file_safe_name(rare_map)
        background_path = backgrounds_dir / f"{background_name}.png"
        if not background_path.exists():
            print(f"Skipping pattern {pattern_id}: missing background {background_path.name}")
            return None

        bg = Image.open(background_path).convert("RGBA")
        if bg.size != (IMAGE_W, IMAGE_H):
            bg = bg.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
    else:
        bg = Image.new("RGBA", (IMAGE_W, IMAGE_H), (0, 0, 0, 0))

    draw = ImageDraw.Draw(bg)
    label_queue = []
    icon_queue = []
    night_icon_queue = []
    landmarks = []
    title = f"Pattern {pattern_id} - {rare_map}"
    if (not under_target_only) and (not HIDE_ALL_BOSSES_AND_ALL_TEXT):
        label_queue.append((20, 40, title, title_font, "title", pattern_id))

    if not under_target_only:
        render_fixed_points(
            icon_queue=icon_queue,
            label_queue=label_queue,
            rare_map=rare_map,
            include_resonating_crystal_points=bool(include_resonating_crystal_points),
        )
        render_great_hollow_hidden_features(
            icon_queue=icon_queue,
            label_queue=label_queue,
            pattern_id=pattern_id,
            rare_map=rare_map,
        )
    for row in entry.get("rows", []):
        if under_target_only and not is_great_hollow_under_recognition_slot_row(
            rare_map=rare_map,
            icon_name=row.get("smallBaseLocationName", ""),
            x_world=row.get("X", ""),
            z_world=row.get("Z", ""),
        ):
            continue
        render_row_item(
            icon_queue=icon_queue,
            row=row,
            label_queue=label_queue,
            rare_map=rare_map,
            render_under_variant=render_under_variant,
        )

    if not under_target_only:
        boss_icon_queue = icon_queue if render_icons else (night_icon_queue if render_night_location_icons else None)
        render_boss(
            icon_queue=boss_icon_queue,
            icon_img=boss_icon,
            boss_name=format_night_boss_text(
                night_index=1,
                boss_name=entry.get("bossId1Name", ""),
                extra_boss_name=entry.get("extraBossId1Name", ""),
            ),
            x_world=entry.get("playArea1_X", ""),
            z_world=entry.get("playArea1_Z", ""),
            label_queue=label_queue,
            label_dx=0,
            label_dy=0,
            render_layer=200,
        )
        render_boss(
            icon_queue=boss_icon_queue,
            icon_img=boss_icon,
            boss_name=format_night_boss_text(
                night_index=2,
                boss_name=entry.get("bossId2Name", ""),
                extra_boss_name=entry.get("extraBossId2Name", ""),
            ),
            x_world=entry.get("playArea2_X", ""),
            z_world=entry.get("playArea2_Z", ""),
            label_queue=label_queue,
            label_dx=0,
            label_dy=0,
            render_layer=200,
        )

    if render_icons:
        for _, _, px, py, icon_name, text, source, source_id in icon_queue:
            append_landmark(
                landmarks,
                landmark_type=icon_name,
                icon=icon_name,
                x=px,
                y=py,
                width=IMAGE_W,
                height=IMAGE_H,
                text=text,
                source=source,
                source_id=source_id,
            )
    elif render_night_location_icons or render_resonating_crystal_icons:
        overlay_icon_landmarks = []
        if render_night_location_icons:
            overlay_icon_landmarks.extend(night_icon_queue)
        if render_resonating_crystal_icons:
            overlay_icon_landmarks.extend(
                item
                for item in icon_queue
                if str(item[6]).strip() == "fixed_point"
                and str(item[4]).strip() == RESONATING_CRYSTAL_OFF_ICON_NAME
            )
        for _, _, px, py, icon_name, text, source, source_id in overlay_icon_landmarks:
            append_landmark(
                landmarks,
                landmark_type=icon_name,
                icon=icon_name,
                x=px,
                y=py,
                width=IMAGE_W,
                height=IMAGE_H,
                text=text,
                source=source,
                source_id=source_id,
            )
    if render_under_variant:
        append_great_hollow_under_recognition_slot_landmarks(
            landmarks=landmarks,
            rare_map=rare_map,
            width=IMAGE_W,
            height=IMAGE_H,
        )

    if render_icons:
        for _, icon_img, px, py, _, _, _, _ in sorted(icon_queue, key=lambda item: item[0]):
            paste_centered(bg, icon_img, px, py)
    elif render_night_location_icons or render_resonating_crystal_icons:
        draw_queue = []
        if render_night_location_icons:
            draw_queue.extend(night_icon_queue)
        if render_resonating_crystal_icons:
            draw_queue.extend(
                item
                for item in icon_queue
                if str(item[6]).strip() == "fixed_point"
                and str(item[4]).strip() == RESONATING_CRYSTAL_OFF_ICON_NAME
            )
        for _, icon_img, px, py, _, _, _, _ in sorted(draw_queue, key=lambda item: item[0]):
            paste_centered(bg, icon_img, px, py)

    if render_labels and RENDER_ANY_TEXT and not HIDE_ALL_BOSSES_AND_ALL_TEXT:
        placed_label_boxes = []
        seen_labels = set()
        final_boss_box = None
        final_boss_name = (
            load_final_boss_names_by_pattern().get(int(pattern_id))
            if not under_target_only else None
        )
        if final_boss_name:
            # Position inside the retained map, including Great Hollow's crop.
            inset = int(CROP_BORDER_BY_RAREMAP.get(rare_map, 0)) + 16
            text = f"Day 3: {final_boss_name}"
            draw.text(
                (inset, inset), text, font=boss_font, anchor="lt",
                fill="white", stroke_fill="black", stroke_width=1,
            )
            final_boss_box = draw.textbbox(
                (inset, inset), text, font=boss_font, anchor="lt", stroke_width=1,
            )
            placed_label_boxes.append(final_boss_box)
            append_landmark(
                landmarks, landmark_type="text", x=inset, y=inset,
                width=IMAGE_W, height=IMAGE_H, text=text,
                source="final_boss", source_id=pattern_id,
            )

        for item in label_queue:
            if len(item) == 6:
                x, y, text, font, label_type, source_id = item
            else:
                x, y, text = item
                font = boss_font
                label_type = "unknown"
                source_id = None

            align = "left" if label_type == "title" else "center"
            if label_type == "title" and final_boss_box is not None:
                # Keep the existing pattern title below the fixed boss heading.
                _, title_box = get_label_bbox(draw, 0, 0, text, font, align="left")
                x = inset
                y = final_boss_box[3] + 12 + (title_box[3] - title_box[1]) / 2

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
                align=align,
            )

            draw_label(draw, final_x, final_y, text, font, align=align)
            placed_label_boxes.append(final_bbox)

            append_landmark(
                landmarks,
                landmark_type="text",
                x=final_x,
                y=final_y,
                width=IMAGE_W,
                height=IMAGE_H,
                text=text,
                source=label_type,
                source_id=source_id,
            )

    crop_px = int(CROP_BORDER_BY_RAREMAP.get(rare_map, 0))
    if crop_px > 0:
        if crop_px * 2 >= bg.width or crop_px * 2 >= bg.height:
            raise ValueError(
                f"Crop for rareMap '{rare_map}' is too large: {crop_px}px for image size {bg.width}x{bg.height}"
            )

        bg = bg.crop((crop_px, crop_px, bg.width - crop_px, bg.height - crop_px))
        apply_crop_to_landmarks(landmarks, crop_px, bg.width, bg.height)

    annotation = build_annotation(pattern_id, rare_map, bg.width, bg.height, landmarks, crop_px)
    return bg, annotation


def build_pattern_image(entry, background_name_override=None, render_under_variant=False):
    result = build_pattern_image_and_annotation(
        entry,
        background_name_override=background_name_override,
        render_under_variant=render_under_variant,
    )
    return result[0] if result is not None else None


def build_pattern_text_overlay_image(
    entry,
    background_name_override=None,
    render_under_variant=False,
    include_night_location_icons=False,
    include_resonating_crystal_icons=False,
):
    result = build_pattern_image_and_annotation(
        entry,
        background_name_override=background_name_override,
        render_under_variant=render_under_variant,
        render_background=False,
        render_icons=False,
        render_labels=True,
        render_night_location_icons=bool(include_night_location_icons),
        render_resonating_crystal_icons=bool(include_resonating_crystal_icons),
        include_resonating_crystal_points=bool(include_resonating_crystal_icons),
    )
    return result[0] if result is not None else None


def render_pattern_entry(
    entry,
    output_dir_target,
    recognizer_display_dir_target,
    recognition_layout_meta_target=None,
    background_name_override=None,
    file_suffix="",
    render_under_variant=False,
):
    image_and_annotation = build_pattern_image_and_annotation(
        entry,
        background_name_override=background_name_override,
        render_under_variant=render_under_variant,
    )
    if image_and_annotation is None:
        return False

    bg, annotation = image_and_annotation
    pattern_id = entry.get("patternId", "")
    rare_map = str(entry.get("rareMap", "")).strip()
    recognizer_display_path = recognizer_display_dir_target / f"layout_{int(pattern_id):04d}.png"
    bg.save(recognizer_display_path)
    print(f"Created: {recognizer_display_path}")

    suffix = str(file_suffix).strip()
    json_path_out = output_dir_target / f"pattern_{int(pattern_id):04d}_{rare_map}{suffix}.json"
    with open(json_path_out, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    print(f"Created: {json_path_out}")

    if recognition_layout_meta_target is not None:
        recognition_layout_meta_target.append(
            collect_pattern_recognition_metadata(
                pattern_id=pattern_id,
                rare_map=rare_map,
                annotation=annotation,
            )
        )
    return True


if __name__ == "__main__":
    run_start_time = time.perf_counter()
    recognition_layout_meta = []
    for entry in data:
        rendered = False
        if not RENDER_ONLY_UNDER_VARIANTS:
            rendered = render_pattern_entry(
                entry=entry,
                output_dir_target=output_dir,
                recognizer_display_dir_target=recognizer_display_dir,
                recognition_layout_meta_target=recognition_layout_meta,
                background_name_override=None,
                file_suffix="",
                render_under_variant=False,
            )

        if UNDER_VARIANT_RENDER_ENABLED and has_great_hollow_under_variant(entry):
            if RENDER_ONLY_UNDER_VARIANTS or rendered:
                render_pattern_entry(
                    entry=entry,
                    output_dir_target=under_output_dir,
                    recognizer_display_dir_target=under_recognizer_display_dir,
                    recognition_layout_meta_target=None,
                    background_name_override=GREAT_HOLLOW_UNDER_BACKGROUND_NAME,
                    file_suffix="_under",
                    render_under_variant=True,
                )

    with open(missing_icons_txt, "w", encoding="utf-8") as f:
        for name in sorted(MISSING_ICON_FILES):
            f.write(f"{name}\n")

    with open(used_icons_txt, "w", encoding="utf-8") as f:
        for name in sorted(USED_ICON_FILES):
            f.write(f"{name}\n")

    export_recognition_assets(recognition_layout_meta)

    print(f"Saved missing icon list: {missing_icons_txt}")
    print(f"Saved used icon list: {used_icons_txt}")
    print(f"Under variant output dir: {under_output_dir}")
    print(f"Under variant recognizer dir: {under_recognizer_display_dir}")
    print(f"Full run took {time.perf_counter() - run_start_time:.2f}s")

print("Done.")
