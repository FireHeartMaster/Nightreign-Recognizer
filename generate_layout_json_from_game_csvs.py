#!/usr/bin/env python3
"""
Generate Nightreign Router-style layout_XXX.json files from game CSVs.

What this script does
---------------------
- Uses LotResultPlayAreaParam.ID as the layout number.
- Uses LotResultSmallBaseAndSpot rows with matching patternId as the final placed content.
- Uses attachId -> SmallBaseAndSpotAttachPoint for the slot ID.
- Uses smallBaseMapId + variationId to resolve the chosen content text.
- Uses only the game's CSVs for structure/selection logic.
- Uses the user-confirmed smallBaseMapId->name text mapping for known IDs.
- For unknown smallBaseMapId values, falls back to the number as text.

Important limitation
--------------------
The stock renderer expects human-readable location strings for Spawn Point / Night Circles /
Merchants and for dynamic POI placement. The game's CSVs do not contain the same final
frontend naming layer. So this script emits ID-based location labels by default, e.g.:
- Spawn Point: "706"
- Night 1 Circle: "1003"
- Major Base location: "100"

These layout files are structurally valid and preserve raw IDs, but to make the stock
renderer fully understand those locations you may still need to extend the renderer's
location mapping (poiLocationMapping.ts or equivalent) to accept these ID strings.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------
# User-confirmed / known names
# ----------------------------

# Boss IDs used directly in LotResultPlayAreaParam (night bosses / extra bosses)
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
    4990: "Ulcerated Tree Spirit",
    # Great Hollow normalized families
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
EVERGAOL_FAMILY_IDS = {4650, 4660, 4670}
CASTLE_IDS = {4941, 4942, 4943}

MAJOR_PREFIXES = ("Fort -", "Camp -", "Ruins -", "Great Church -", "Castle -")
MINOR_PREFIXES = ("Church -", "Sorcerer's Rise", "Township -")

SHIFTING_EARTH_BY_MODIFIER = {
    11: "Mountaintop",
    12: "The Crater",
    13: "Rotted Woods",
    14: "Great Hollow",
    15: "Shrouded City",
}


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: object, default: Optional[int] = None) -> Optional[int]:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def load_known_smallbase_names(path: Path) -> Dict[int, str]:
    """
    Reads the two-column no-header CSV previously created in this chat:
      mapId,"name text"
    """
    names: Dict[int, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    map_id = int(row[0])
                except Exception:
                    continue
                text = row[1] if len(row) > 1 else str(map_id)
                names[map_id] = text
    # User later confirmed this explicitly.
    names[5110] = "Sorcerer's Rise"
    return names


def strip_category_prefix(text: str) -> str:
    if " - " in text:
        return text.split(" - ", 1)[1].strip()
    return text.strip()


def resolve_smallbase_name(raw_id: int, variation_id: int, known_names: Dict[int, str]) -> str:
    """
    Resolve a smallBaseMapId into the most specific text we currently know.

    Rules:
    - If unknown: return the number as text.
    - If known and has semicolon-separated options: use variationId as 0-based selector
      when it fits the number of options.
    - Otherwise return the known text as-is.
    """
    base = known_names.get(raw_id)
    if base is None:
        return str(raw_id)

    # Example:
    #   "Fort - Guardian Golem; Crystalians"
    #   "Field Boss - Red Wolf"
    #   "Sorcerer's Rise - Normal"
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


def normalized_value_for_layout(category: str, resolved_text: str) -> str:
    """
    Make the value closer to what the renderer usually expects.
    - Boss-like values: strip the category prefix so the value is just the boss name.
    - Event values: keep as "Map Event - X"
    - Major/minor landmark values: keep the category prefix (Fort - X, Church - X, etc.)
    """
    if category in {"Field Boss", "Field Boss Red", "Arena Boss", "Evergaol", "Castle"}:
        return strip_category_prefix(resolved_text)

    if category == "Map Event":
        core = strip_category_prefix(resolved_text)
        if core.startswith("Map Event -"):
            return core
        return f"Map Event - {core}"

    return resolved_text


def determine_shifting_earth(pattern_id: int, flags_by_pattern: Dict[int, List[dict]]) -> str:
    mods = {as_int(r.get("modifier"), -999) for r in flags_by_pattern.get(pattern_id, [])}
    for mod, name in SHIFTING_EARTH_BY_MODIFIER.items():
        if mod in mods:
            return name
    return "Default"


def determine_spawn_point(pattern_id: int, flags_by_pattern: Dict[int, List[dict]]) -> str:
    """
    CSV-only / ID-first:
    - base/default side usually uses modifierSet 190 -> 700..708
    - Great Hollow side can use 13000..13002
    Return the modifier as a string, or "empty" if absent.
    """
    spawn_modifiers: List[int] = []
    for row in flags_by_pattern.get(pattern_id, []):
        mset = as_int(row.get("modifierSet"), -1)
        mod = as_int(row.get("modifier"), -1)
        if mset in {160, 190} and mod > 0:
            spawn_modifiers.append(mod)
    if not spawn_modifiers:
        return "empty"
    # Prefer the last nonzero if multiple exist.
    return str(spawn_modifiers[-1])


def classify_attach_row(attach_id: int, smallbase_map_id: int, resolved_text: str) -> str:
    if attach_id == 190 or smallbase_map_id in CASTLE_IDS:
        return "Castle"
    if attach_id in {756, 757}:
        return "Arena Boss"
    if 750 <= attach_id <= 763:
        if smallbase_map_id in FIELD_BOSS_RED_IDS:
            return "Field Boss Red"
        if smallbase_map_id in BASEMENT_BOSS_IDS:
            return "Arena Boss"
        return "Field Boss"
    if 601 <= attach_id <= 607 or smallbase_map_id in EVERGAOL_FAMILY_IDS:
        return "Evergaol"
    if smallbase_map_id in EVENT_IDS or 520 <= attach_id <= 596 or 800 <= attach_id <= 817:
        return "Map Event"
    if 710 <= attach_id <= 718:
        return "Scale-Bearing Merchant"
    if 100 <= attach_id < 200 or 1100 <= attach_id < 1300 or 2000 <= attach_id < 2200:
        return "Major Base"
    if 300 <= attach_id < 400 or 2300 <= attach_id < 2400:
        return "Minor Base"

    if resolved_text.startswith(MAJOR_PREFIXES):
        return "Major Base"
    if resolved_text.startswith(MINOR_PREFIXES):
        return "Minor Base"

    return "Other"


def build_entry_key(category: str, attach_id: int) -> Tuple[str, str]:
    """
    Return (key, location_string).
    These location strings are intentionally ID-based because the game's CSVs do not
    provide the same frontend naming layer as the router repo.
    """
    if category == "Castle":
        return "Castle - Castle", "Castle"
    if category == "Scale-Bearing Merchant":
        return "Scale-Bearing Merchant", str(attach_id)
    return f"{category} - {attach_id}", str(attach_id)


def build_layout_for_pattern(
    layout_id: int,
    play_row: dict,
    result_rows_for_pattern: List[dict],
    flags_by_pattern: Dict[int, List[dict]],
    known_names: Dict[int, str],
) -> dict:
    pattern_id = as_int(play_row.get("patternId"), layout_id)
    out: Dict[str, object] = {}
    out["Shifting Earth"] = determine_shifting_earth(pattern_id, flags_by_pattern)

    # Night bosses / circles / spawn point (ID-first, CSV-only)
    boss1 = as_int(play_row.get("bossId1"), -1)
    boss2 = as_int(play_row.get("bossId2"), -1)
    extra1 = as_int(play_row.get("extraBossId1"), -1)
    extra2 = as_int(play_row.get("extraBossId2"), -1)
    pa1 = as_int(play_row.get("playArea1"), -1)
    pa2 = as_int(play_row.get("playArea2"), -1)

    out["Night 1 Boss"] = BOSS_ID_TO_NAME.get(boss1, str(boss1) if boss1 and boss1 >= 0 else "empty")
    out["Night 2 Boss"] = BOSS_ID_TO_NAME.get(boss2, str(boss2) if boss2 and boss2 >= 0 else "empty")
    out["Night 1 Boss ID"] = boss1
    out["Night 2 Boss ID"] = boss2

    if extra1 is not None and extra1 >= 0:
        out["Extra Boss 1"] = BOSS_ID_TO_NAME.get(extra1, str(extra1))
        out["Extra Boss 1 ID"] = extra1
    if extra2 is not None and extra2 >= 0:
        out["Extra Boss 2"] = BOSS_ID_TO_NAME.get(extra2, str(extra2))
        out["Extra Boss 2 ID"] = extra2

    out["Night 1 Circle"] = str(pa1) if pa1 and pa1 >= 0 else "empty"
    out["Night 2 Circle"] = str(pa2) if pa2 and pa2 >= 0 else "empty"
    out["Night 1 Circle ID"] = pa1
    out["Night 2 Circle ID"] = pa2

    out["Spawn Point"] = determine_spawn_point(pattern_id, flags_by_pattern)
    out["Pattern ID"] = pattern_id

    # Defaults for optional router fields
    out["Scale-Bearing Merchant"] = "empty"
    out["Frenzy Tower"] = "empty"
    out["Special Event"] = "empty"

    # Preserve LotResultSmallBaseAndSpot table order
    for row in result_rows_for_pattern:
        attach_id = as_int(row.get("attachId"), -1)
        sb_id = as_int(row.get("smallBaseMapId"), -1)
        variation_id = as_int(row.get("variationId"), 0)
        if attach_id is None or attach_id < 0 or sb_id is None or sb_id < 0:
            continue

        resolved_text = resolve_smallbase_name(sb_id, variation_id, known_names)
        category = classify_attach_row(attach_id, sb_id, resolved_text)

        if category == "Other":
            # Still preserve it, but make the category explicit.
            key, location = f"Other - {attach_id}", str(attach_id)
            value = resolved_text
        elif category == "Scale-Bearing Merchant":
            # Top-level single field expected by renderer
            out["Scale-Bearing Merchant"] = str(attach_id)
            continue
        else:
            key, location = build_entry_key(category, attach_id)
            value = normalized_value_for_layout(category, resolved_text)

        # Promote event row to the pattern-level Special Event field if present.
        if category == "Map Event" and out.get("Special Event") == "empty":
            out["Special Event"] = strip_category_prefix(resolved_text)

        entry = {
            "location": location,
            "value": value,
            "attachId": attach_id,
            "smallBaseMapId": sb_id,
            "variationId": variation_id,
            "mapIndex": as_int(row.get("mapIndex"), 0),
        }
        if key in out:
            # Avoid key collisions by appending the raw row ID.
            row_id = as_int(row.get("ID"), 0)
            key = f"{key} [{row_id}]"
        out[key] = entry

    return out


def parse_pattern_args(values: List[str]) -> List[int]:
    result: List[int] = []
    for value in values:
        if "-" in value:
            a, b = value.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(value))
    return sorted(set(result))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate layout_XXX.json files from Nightreign game CSVs.")
    ap.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing the CSV files.")
    ap.add_argument("--output-dir", type=Path, default=Path("./pattern_layouts_generated"), help="Where to write layout_XXX.json files.")
    ap.add_argument("--patterns", nargs="*", default=[], help="Optional pattern/layout IDs, e.g. 15 1005-1009. Default: all rows in LotResultPlayAreaParam.")
    ap.add_argument("--known-mapid-csv", type=Path, default=None, help="Optional two-column no-header CSV with mapId,name text mappings.")
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lot_result_play = read_csv(input_dir / "LotResultPlayAreaParam.csv")
    lot_result_small = read_csv(input_dir / "LotResultSmallBaseAndSpot.csv")
    lot_result_flags = read_csv(input_dir / "LotResultMapPatternFlag.csv")

    known_mapid_csv = args.known_mapid_csv
    if known_mapid_csv is None:
        candidate = input_dir / "smallbaseandspotdefine_mapid_name_only.csv"
        known_mapid_csv = candidate if candidate.exists() else candidate
    known_names = load_known_smallbase_names(known_mapid_csv)

    results_by_pattern: Dict[int, List[dict]] = defaultdict(list)
    for row in lot_result_small:
        pid = as_int(row.get("patternId"), -1)
        if pid is not None and pid >= 0:
            results_by_pattern[pid].append(row)

    flags_by_pattern: Dict[int, List[dict]] = defaultdict(list)
    for row in lot_result_flags:
        pid = as_int(row.get("patternId"), -1)
        if pid is not None and pid >= 0:
            flags_by_pattern[pid].append(row)

    play_rows_by_layout_id: Dict[int, dict] = {}
    for row in lot_result_play:
        layout_id = as_int(row.get("ID"), -1)
        if layout_id is not None and layout_id >= 0:
            play_rows_by_layout_id[layout_id] = row

    if args.patterns:
        selected_ids = parse_pattern_args(args.patterns)
    else:
        selected_ids = sorted(play_rows_by_layout_id.keys())

    written = 0
    for layout_id in selected_ids:
        play_row = play_rows_by_layout_id.get(layout_id)
        if play_row is None:
            continue
        pattern_id = as_int(play_row.get("patternId"), layout_id)
        layout = build_layout_for_pattern(
            layout_id=layout_id,
            play_row=play_row,
            result_rows_for_pattern=results_by_pattern.get(pattern_id, []),
            flags_by_pattern=flags_by_pattern,
            known_names=known_names,
        )
        out_path = output_dir / f"layout_{layout_id:03d}.json"
        out_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(f"Wrote {written} layout file(s) to {output_dir}")


if __name__ == "__main__":
    main()
