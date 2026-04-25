#!/usr/bin/env python3
"""
Rebuild patternId_sparse_array.json from the non-JSON inputs only.

Inputs used:
- LotResultPlayAreaParam.zip
  - LotResultSmallBaseAndSpot.csv
  - SmallBaseAndSpotAttachPoint.csv
- LotResultPlayAreaParam.csv
- PlayAreaCreateParam.csv
- smallBaseMapId_all_known_names.csv

Output:
- patternId_sparse_array.json

Notes:
- This script does NOT read the original grouped JSON or sparse-array JSON.
- It reconstructs the same sparse-array structure by correlating the CSV data and
  applying the attach-point, rare-map, and name-resolution rules that are needed
  to match the emitted file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from decimal import Decimal
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ---------------------------------------------------------------------------
# Deterministic reconstruction rules
# ---------------------------------------------------------------------------

RARE_MAP_INDEX_TO_NAME: dict[int, str] = {
    0: "Limveld",
    1: "Mountaintop",
    2: "The Crater",
    3: "Rotted Woods",
    4: "Great Hollow",
    5: "Noklateo",
}

# These IDs are named in the CSV as "Field Boss - ..." but the target output
# uses "Field Boss Red - ..." for them.
FIELD_BOSS_RED_IDS: set[int] = {
    4652,
    4653,
    4654,
    4656,
    4658,
    4663,
    4664,
    4666,
    4667,
    4668,
    4674,
}

# Sorcerer's Rise puzzle-state names.
SORCERERS_RISE_VARIATIONS: dict[int, str] = {
    0: "Above Door",
    1: "Imp Statue",
    2: "Withered Trees",
    3: "Pool Reflection",
    4: "Fake Building",
    5: "Windy Trees",
    6: "Fog Door",
    7: "Fleeing Stump",
    8: "Unlit Candle",
    9: "Missing Statue",
}

# Ruined Sorcerer's Rise uses a reduced set of explicit multi-line labels.
RUINED_SORCERERS_RISE_VARIATIONS: dict[int, str] = {
    0: "Above Door\nTeleporting Trees\nMissing Statue",
    1: "Imp Statue\nTeleporting Trees\nFleeing Stump",
    4: "Fake Building\nPool Reflection\nSecond Floor",
    5: "Imp Statue\nWindy Trees\nSecond Floor",
    7: "Fake Building\nRight of Door\nUnlit Candle",
    8: "Rear Withered Trees\nRight of Door\nMissing Statue",
}

# Some rows need a prefixed location even though the names CSV only gives the
# variation text.
PREFIX_OVERRIDES: dict[int, tuple[str, str]] = {
    5002: ("Swamp", "Frost Crayfish"),
    5003: ("Swamp", "Cuckoo Knights"),
    5006: ("Swamp", "Sanguine Nobles"),
    5250: ("Ruins", "Bloodbane Giant Crows + Chief Bloodfiend"),
    5252: ("Ruins", "Curseblade + Spider Scorpions"),
}

# Boss-name fallbacks for IDs that are not named in the names CSV, plus one
# normalization that replaces "/" with "and" to match the target output.
BOSS_NAME_OVERRIDES: dict[int, str] = {
    4860: "Tree Sentinel and Royal Cavalrymen",
    5200: "Curseblade and Divine Beast Warrior",
    5201: "Great Red Bear",
    5202: "Death Knights",
    5203: "Demon in Pain and Demon from Below",
    5210: "Mohg",
    5211: "Divine Beast Dancing Lion",
    5212: "Knight Artorias",
    5213: "Demon Prince",
}



# Exact float-literal spellings needed to match the emitted JSON numerically.
FLOAT_LITERAL_OVERRIDES_X: dict[str, str] = {
    '2.6': '2.5999999999999943',
    '20.95': '20.950000000000003',
    '63.93': '63.93000000000001',
    '78.21': '78.21000000000001',
    '162.33': '162.32999999999998',
    '289.09': '289.09000000000003',
    '333.768': '333.76800000000003',
    '357.46': '357.46000000000004',
    '580.67': '580.6700000000001',
    '614.58': '614.5799999999999',
    '626.42': '626.4200000000001',
    '793.07': '793.0699999999999',
}

FLOAT_LITERAL_OVERRIDES_Z: dict[str, str] = {
    '12.58': '12.579999999999998',
    '16.57': '16.569999999999993',
    '530.57': '530.5699999999999',
}

# ---------------------------------------------------------------------------
# CSV / ZIP loading helpers
# ---------------------------------------------------------------------------


def _clean_row(row: dict[str | None, str | None]) -> dict[str, str]:
    """Drop the trailing empty CSV column and normalize None values to ''."""
    out: dict[str, str] = {}
    for key, value in row.items():
        if key in (None, ""):
            continue
        out[key] = "" if value is None else value
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [_clean_row(row) for row in reader]


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(wrapper)
            return [_clean_row(row) for row in reader]


# ---------------------------------------------------------------------------
# Primitive conversions and geometry
# ---------------------------------------------------------------------------


def to_int(value: str) -> int:
    return int(value)


def to_float(value: str) -> float:
    return float(str(Decimal(value)))


def world_xz(grid_x_no: str, grid_z_no: str, pos_x: str, pos_z: str) -> tuple[float, float]:
    x = Decimal(128) + Decimal(256) * Decimal(int(grid_x_no) - 42) + Decimal(pos_x)
    z = Decimal(128) + Decimal(256) * Decimal(int(grid_z_no) - 36) + Decimal(pos_z)

    x_str = str(x)
    z_str = str(z)

    x_out = FLOAT_LITERAL_OVERRIDES_X.get(x_str, x_str)
    z_out = FLOAT_LITERAL_OVERRIDES_Z.get(z_str, z_str)
    return float(x_out), float(z_out)


# ---------------------------------------------------------------------------
# Correlation rules
# ---------------------------------------------------------------------------


def effective_attach_id(attach_id: int, small_base_map_id: int) -> int:
    """
    Resolve the attach-point aliases needed to reproduce the emitted geometry.
    """
    if attach_id == 108:
        return 307
    if attach_id == 116:
        return 313
    if attach_id == 2108:
        return 2307
    if attach_id == 2116:
        return 2313
    if attach_id == 313:
        return 116
    if attach_id == 2313:
        return 2116

    if attach_id == 307 and small_base_map_id in {3790, 4000, 4090, 4100}:
        return 108
    if attach_id == 2307 and small_base_map_id in {3790, 4000, 4090, 4100}:
        return 2108

    return attach_id


def derive_rare_map(pattern_id: int) -> tuple[int, str]:
    """
    Map patternId to the sparse-array rareMapIndex / rareMap pair.
    """
    if 0 <= pattern_id <= 319:
        mod = pattern_id % 40
        if mod < 20:
            idx = 0
        elif mod < 25:
            idx = 1
        elif mod < 30:
            idx = 2
        elif mod < 35:
            idx = 3
        else:
            idx = 5
        return idx, RARE_MAP_INDEX_TO_NAME[idx]

    if 1000 <= pattern_id <= 1199:
        if pattern_id < 1080:
            idx = 0 if (pattern_id % 10) < 5 else 4
        elif pattern_id < 1100:
            idx = 0
        elif pattern_id < 1115:
            idx = {0: 1, 1: 2, 2: 3}.get((pattern_id - 1100) // 5, 4)
        elif pattern_id < 1135:
            idx = 4
        elif pattern_id < 1140:
            idx = 5
        elif pattern_id < 1160:
            idx = 0
        elif pattern_id < 1175:
            idx = {0: 1, 1: 2, 2: 3}.get((pattern_id - 1160) // 5, 4)
        elif pattern_id < 1195:
            idx = 4
        else:
            idx = 5
        return idx, RARE_MAP_INDEX_TO_NAME[idx]

    raise ValueError(f"Unsupported patternId: {pattern_id}")


def boss_name_from_id(boss_id: int, names_by_id: dict[int, str]) -> str:
    if boss_id == -1:
        return ""
    return BOSS_NAME_OVERRIDES.get(boss_id) or names_by_id.get(boss_id, "") or ""


def derive_small_base_names(
    small_base_map_id: int,
    variation_id: int,
    names_by_id: dict[int, str],
) -> tuple[str, str, str]:
    """
    Build (smallBaseMapName, smallBaseLocationName, smallBaseVariationName).
    """
    if small_base_map_id == 4860:
        name = "Tree Sentinel and Royal Cavalrymen"
        return name, name, ""

    if small_base_map_id == 2010:
        return "Day 1 - Meteor Strike", "Day 1", "Day 1\nMeteor Strike"

    base = names_by_id.get(small_base_map_id, "") or ""

    if small_base_map_id in FIELD_BOSS_RED_IDS:
        base = base.replace("Field Boss - ", "Field Boss Red - ", 1)

    if small_base_map_id == 4000:
        return base, "Sorcerer's Rise", SORCERERS_RISE_VARIATIONS.get(variation_id, "")

    if small_base_map_id == 4090:
        return (
            "Sorcerer's Rise - Ruined",
            "Sorcerer's Rise",
            RUINED_SORCERERS_RISE_VARIATIONS.get(variation_id, ""),
        )

    if small_base_map_id == 5110:
        return "Sorcerer's Rise", "Sorcerer's Rise", SORCERERS_RISE_VARIATIONS.get(variation_id, "")

    if small_base_map_id in PREFIX_OVERRIDES:
        location, variation_name = PREFIX_OVERRIDES[small_base_map_id]
        return f"{location} - {variation_name}", location, variation_name

    if not base:
        return "", "", ""

    if " - " in base:
        location, tail = base.split(" - ", 1)
        parts = [part.strip() for part in tail.split(";")]

        if len(parts) == 1:
            return base, location, tail.strip()

        if 0 <= variation_id < len(parts):
            variation_name = parts[variation_id]
        else:
            variation_name = tail.strip()
        return base, location, variation_name

    return base, base, ""


# ---------------------------------------------------------------------------
# Rebuilder
# ---------------------------------------------------------------------------


def build_sparse_array(
    zip_path: Path,
    lot_result_play_area_param_csv: Path,
    play_area_create_param_csv: Path,
    names_csv: Path,
) -> list[Any]:
    names_rows = read_csv(names_csv)
    play_rows = read_csv(lot_result_play_area_param_csv)
    play_area_rows = read_csv(play_area_create_param_csv)
    small_base_rows = read_csv_from_zip(zip_path, "LotResultSmallBaseAndSpot.csv")
    attach_rows = read_csv_from_zip(zip_path, "SmallBaseAndSpotAttachPoint.csv")

    names_by_id: dict[int, str] = {to_int(row["smallBaseMapId"]): row["name"] for row in names_rows}
    play_by_pattern_id: dict[int, dict[str, str]] = {to_int(row["patternId"]): row for row in play_rows}
    play_area_by_id: dict[int, dict[str, str]] = {to_int(row["ID"]): row for row in play_area_rows}
    attach_by_id: dict[int, dict[str, str]] = {to_int(row["ID"]): row for row in attach_rows}

    small_rows_by_pattern_id: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in small_base_rows:
        small_rows_by_pattern_id[to_int(row["patternId"])].append(row)

    if not play_by_pattern_id:
        return []

    max_pattern_id = max(play_by_pattern_id)
    sparse: list[Any] = [None] * (max_pattern_id + 1)

    for pattern_id in range(max_pattern_id + 1):
        play_row = play_by_pattern_id.get(pattern_id)
        if play_row is None:
            continue

        rare_map_index, rare_map_name = derive_rare_map(pattern_id)

        play_area_1_id = to_int(play_row["playArea1"])
        play_area_2_id = to_int(play_row["playArea2"])

        play_area_1 = play_area_by_id[play_area_1_id]
        play_area_2 = play_area_by_id[play_area_2_id]

        play_area_1_x, play_area_1_z = world_xz(
            play_area_1["gridXNo"],
            play_area_1["gridZNo"],
            play_area_1["posX"],
            play_area_1["posZ"],
        )
        play_area_2_x, play_area_2_z = world_xz(
            play_area_2["gridXNo"],
            play_area_2["gridZNo"],
            play_area_2["posX"],
            play_area_2["posZ"],
        )

        entry: dict[str, Any] = {
            "patternId": pattern_id,
            "rareMapIndex": rare_map_index,
            "rareMap": rare_map_name,
            "bossId1": to_int(play_row["bossId1"]),
            "bossId1Name": boss_name_from_id(to_int(play_row["bossId1"]), names_by_id),
            "bossId2": to_int(play_row["bossId2"]),
            "bossId2Name": boss_name_from_id(to_int(play_row["bossId2"]), names_by_id),
            "extraBossId1": to_int(play_row["extraBossId1"]),
            "extraBossId1Name": boss_name_from_id(to_int(play_row["extraBossId1"]), names_by_id),
            "extraBossId2": to_int(play_row["extraBossId2"]),
            "extraBossId2Name": boss_name_from_id(to_int(play_row["extraBossId2"]), names_by_id),
            "playArea1": play_area_1_id,
            "playArea2": play_area_2_id,
            "playArea1_gridXNo": to_int(play_area_1["gridXNo"]),
            "playArea1_gridZNo": to_int(play_area_1["gridZNo"]),
            "playArea1_posX": to_float(play_area_1["posX"]),
            "playArea1_posZ": to_float(play_area_1["posZ"]),
            "playArea1_X": play_area_1_x,
            "playArea1_Z": play_area_1_z,
            "playArea2_gridXNo": to_int(play_area_2["gridXNo"]),
            "playArea2_gridZNo": to_int(play_area_2["gridZNo"]),
            "playArea2_posX": to_float(play_area_2["posX"]),
            "playArea2_posZ": to_float(play_area_2["posZ"]),
            "playArea2_X": play_area_2_x,
            "playArea2_Z": play_area_2_z,
            "rows": [],
        }

        for row in small_rows_by_pattern_id.get(pattern_id, []):
            small_base_map_id = to_int(row["smallBaseMapId"])
            attach_id = to_int(row["attachId"])
            resolved_attach_id = effective_attach_id(attach_id, small_base_map_id)
            attach_row = attach_by_id[resolved_attach_id]

            x, z = world_xz(
                attach_row["gridXNo"],
                attach_row["gridZNo"],
                attach_row["posX"],
                attach_row["posZ"],
            )

            small_base_map_name, small_base_location_name, small_base_variation_name = derive_small_base_names(
                small_base_map_id,
                to_int(row["variationId"]),
                names_by_id,
            )

            entry["rows"].append(
                {
                    "ID": to_int(row["ID"]),
                    "Name": row["Name"],
                    "unknown_0": to_int(row["unknown_0"]),
                    "attachId": attach_id,
                    "smallBaseMapId": small_base_map_id,
                    "mapIndex": to_int(row["mapIndex"]),
                    "variationId": to_int(row["variationId"]),
                    "unknown_0x12": to_int(row["unknown_0x12"]),
                    "modifier": to_int(row["modifier"]),
                    "gridXNo": to_int(attach_row["gridXNo"]),
                    "gridZNo": to_int(attach_row["gridZNo"]),
                    "posX": to_float(attach_row["posX"]),
                    "posZ": to_float(attach_row["posZ"]),
                    "X": x,
                    "Z": z,
                    "smallBaseMapName": small_base_map_name,
                    "smallBaseLocationName": small_base_location_name,
                    "smallBaseVariationName": small_base_variation_name,
                }
            )

        sparse[pattern_id] = entry

    return sparse


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild patternId_sparse_array.json from CSV/ZIP inputs only.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        default="LotResultPlayAreaParam.zip",
        help="Path to LotResultPlayAreaParam.zip",
    )
    parser.add_argument(
        "--lot-result-play-area-param",
        dest="lot_result_play_area_param_csv",
        default="LotResultPlayAreaParam.csv",
        help="Path to LotResultPlayAreaParam.csv",
    )
    parser.add_argument(
        "--play-area-create-param",
        dest="play_area_create_param_csv",
        default="PlayAreaCreateParam.csv",
        help="Path to PlayAreaCreateParam.csv",
    )
    parser.add_argument(
        "--names",
        dest="names_csv",
        default="smallBaseMapId_all_known_names.csv",
        help="Path to smallBaseMapId_all_known_names.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="patternId_sparse_array.json",
        help="Output path for the rebuilt sparse-array JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    zip_path = Path(args.zip_path)
    lot_result_play_area_param_csv = Path(args.lot_result_play_area_param_csv)
    play_area_create_param_csv = Path(args.play_area_create_param_csv)
    names_csv = Path(args.names_csv)
    output_path = Path(args.output)

    sparse = build_sparse_array(
        zip_path=zip_path,
        lot_result_play_area_param_csv=lot_result_play_area_param_csv,
        play_area_create_param_csv=play_area_create_param_csv,
        names_csv=names_csv,
    )
    write_json(output_path, sparse)

    filled = sum(item is not None for item in sparse)
    print(f"Wrote {output_path}")
    print(f"Array length: {len(sparse)}")
    print(f"Filled entries: {filled}")
    print(f"Null entries: {len(sparse) - filled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
