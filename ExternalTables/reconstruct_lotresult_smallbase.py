#!/usr/bin/env python3
"""
Reconstruct LotResultSmallBaseAndSpot_grouped_by_patternId_14.json from the CSVs in
LotResultPlayAreaParam.zip.

Inputs expected inside the zip:
  - LotResultPlayAreaParam.csv
  - LotResultSmallBaseAndSpot.csv
  - MapPatternSet.csv
  - SmallBaseAndSpotAttachPoint.csv

What this script rebuilds:
  - the patternId-grouped JSON structure
  - the row-level coordinates from SmallBaseAndSpotAttachPoint.csv
  - boss names for the 8 ids you supplied manually in chat
  - playArea1 / playArea2 IDs and, when possible, their coordinates

Important note:
  The CSVs clearly provide patternId, boss modifiers, row attachments and attach points.
  The exact playArea coordinate anchors are not explicitly labeled in these exports, so
  the script uses a best-effort resolver:
    1) direct attach-point lookup when the playArea id matches an attach-point ID
    2) a patternId-based fallback that looks for a unique blank/placeholder anchor row
       inside the referenced play-area pattern
  If your export uses a different anchor rule, only the `_resolve_play_area_coords`
  function should need adjustment.

Usage:
  python reconstruct_lotresult_smallbase.py /path/to/LotResultPlayAreaParam.zip \
      -o LotResultSmallBaseAndSpot_grouped_by_patternId_14_reconstructed.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --- Manual boss-name mappings supplied in chat ---
BOSS_ID_TO_NAME = {
    5200: "Curseblade and Divine Beast Warrior",
    5201: "Great Red Bear",
    5202: "Death Knights",
    5203: "Demon in Pain and Demon from Below",
    5210: "Mohg",
    5211: "Divine Beast Dancing Lion",
    5212: "Knight Artorias",
    5213: "Demon Prince",
}

# If you want to extend boss naming later, add ids here.
# Unknown ids will keep an empty string in the output, matching the source style.

RARE_MAP_NAME = "Limveld"
RARE_MAP_INDEX = 0


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> List[Dict[str, str]]:
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        return list(reader)


def to_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def calc_world_x(grid_x: int, pos_x: float) -> float:
    # Derived from the exported JSON examples:
    # X = (gridXNo - 42) * 256 + posX + 128
    return (grid_x - 42) * 256 + pos_x + 128.0


def calc_world_z(grid_z: int, pos_z: float) -> float:
    # Derived from the exported JSON examples:
    # Z = (gridZNo - 36) * 256 + posZ + 128
    return (grid_z - 36) * 256 + pos_z + 128.0


def is_blank_row(row: Dict[str, Any]) -> bool:
    return (
        str(row.get("smallBaseMapName", "")).strip() == ""
        and str(row.get("smallBaseLocationName", "")).strip() == ""
        and str(row.get("smallBaseVariationName", "")).strip() == ""
    )


def normalize_smallbase_row(r: Dict[str, str]) -> Dict[str, Any]:
    out = {
        "ID": to_int(r.get("ID")),
        "Name": r.get("Name", "") or "",
        "unknown_0": to_int(r.get("unknown_0")),
        "patternId": to_int(r.get("patternId")),
        "attachId": to_int(r.get("attachId")),
        "smallBaseMapId": to_int(r.get("smallBaseMapId")),
        "mapIndex": to_int(r.get("mapIndex")),
        "variationId": to_int(r.get("variationId")),
        "unknown_0x12": to_int(r.get("unknown_0x12")),
        "modifier": to_int(r.get("modifier")),
        "gridXNo": None,
        "gridZNo": None,
        "posX": None,
        "posZ": None,
        "X": None,
        "Z": None,
        "smallBaseMapName": "",
        "smallBaseLocationName": "",
        "smallBaseVariationName": "",
    }
    return out


def normalize_attach_point(r: Dict[str, str]) -> Dict[str, Any]:
    out = {
        "ID": to_int(r.get("ID")),
        "gridXNo": to_int(r.get("gridXNo")),
        "gridZNo": to_int(r.get("gridZNo")),
        "posX": to_float(r.get("posX")),
        "posY": to_float(r.get("posY")),
        "posZ": to_float(r.get("posZ")),
        "attachPointEntityId": to_int(r.get("attachPointEntityId")),
        "defaultSmallBase": to_int(r.get("defaultSmallBase")),
        "raw": r,
    }
    out["X"] = calc_world_x(out["gridXNo"], out["posX"])
    out["Z"] = calc_world_z(out["gridZNo"], out["posZ"])
    return out


def load_zip_tables(zip_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    with zipfile.ZipFile(zip_path) as zf:
        play = read_csv_from_zip(zf, "LotResultPlayAreaParam.csv")
        small = read_csv_from_zip(zf, "LotResultSmallBaseAndSpot.csv")
        pattern_set = read_csv_from_zip(zf, "MapPatternSet.csv")
        attach = read_csv_from_zip(zf, "SmallBaseAndSpotAttachPoint.csv")
    return play, small, pattern_set, attach


def build_attach_lookup(attach_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, Any]]:
    return {r["ID"]: normalize_attach_point(r) for r in attach_rows}


def build_small_lookup(small_rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in small_rows:
        row = normalize_smallbase_row(r)
        groups[row["patternId"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda x: x["ID"])
    return dict(groups)


def build_play_lookup(play_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {to_int(r.get("patternId")): r for r in play_rows}


def choose_anchor_row(pattern_rows: List[Dict[str, Any]], target_pattern_id: int) -> Optional[Dict[str, Any]]:
    """
    Best-effort resolver for playArea coordinates.

    1) Prefer a row with the 'blank' placeholder names.
    2) If there is no blank row, prefer a row whose attachId is the smallest.
    3) If there are multiple blank rows, prefer the one with the smallest ID.
    """
    if not pattern_rows:
        return None

    blank_rows = [r for r in pattern_rows if is_blank_row(r)]
    if blank_rows:
        blank_rows.sort(key=lambda r: r["ID"])
        return blank_rows[0]

    rows_sorted = sorted(pattern_rows, key=lambda r: r["ID"])
    return rows_sorted[0]


def _resolve_play_area_coords(
    play_area_id: int,
    small_lookup: Dict[int, List[Dict[str, Any]]],
    attach_lookup: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Attempt to resolve play-area coordinates from the CSVs.

    Returns a dict with gridXNo/gridZNo/posX/posZ/X/Z if a reasonable anchor can be found.
    """
    # Direct attach-point lookup (works when the playArea id is literally an attach-point ID)
    if play_area_id in attach_lookup:
        a = attach_lookup[play_area_id]
        return {
            "gridXNo": a["gridXNo"],
            "gridZNo": a["gridZNo"],
            "posX": a["posX"],
            "posZ": a["posZ"],
            "X": a["X"],
            "Z": a["Z"],
        }

    # Fallback: treat the playArea id as a patternId and use a stable anchor row.
    pattern_rows = small_lookup.get(play_area_id, [])
    anchor = choose_anchor_row(pattern_rows, play_area_id)
    if anchor is None:
        return None

    # If the anchor row has a matching attach point, use that exact attach-point row.
    attach_id = anchor.get("attachId", 0)
    if isinstance(attach_id, int) and attach_id in attach_lookup:
        a = attach_lookup[attach_id]
        return {
            "gridXNo": a["gridXNo"],
            "gridZNo": a["gridZNo"],
            "posX": a["posX"],
            "posZ": a["posZ"],
            "X": a["X"],
            "Z": a["Z"],
        }

    # Otherwise use the anchor row itself, if coordinates are present in the CSV.
    if anchor.get("gridXNo") is not None and anchor.get("posX") is not None:
        return {
            "gridXNo": int(anchor["gridXNo"]),
            "gridZNo": int(anchor["gridZNo"]),
            "posX": float(anchor["posX"]),
            "posZ": float(anchor["posZ"]),
            "X": float(anchor["X"]),
            "Z": float(anchor["Z"]),
        }

    return None


def reconstruct(zip_path: Path) -> List[Dict[str, Any]]:
    play_rows, small_rows, _pattern_set_rows, attach_rows = load_zip_tables(zip_path)
    play_lookup = build_play_lookup(play_rows)
    small_lookup = build_small_lookup(small_rows)
    attach_lookup = build_attach_lookup(attach_rows)

    # Group the small-base rows by the top-level patternId from playLookup.
    output: List[Dict[str, Any]] = []

    for pattern_id in sorted(play_lookup.keys()):
        p = play_lookup[pattern_id]

        boss_id1 = to_int(p.get("bossId1"))
        boss_id2 = to_int(p.get("bossId2"))
        extra_boss_id1 = to_int(p.get("extraBossId1"), -1)
        extra_boss_id2 = to_int(p.get("extraBossId2"), -1)

        row_dicts: List[Dict[str, Any]] = []
        for src in small_lookup.get(pattern_id, []):
            attach_id = src["attachId"]
            attach = attach_lookup.get(attach_id)
            row = dict(src)
            if attach is not None:
                row["gridXNo"] = attach["gridXNo"]
                row["gridZNo"] = attach["gridZNo"]
                row["posX"] = attach["posX"]
                row["posZ"] = attach["posZ"]
                row["X"] = attach["X"]
                row["Z"] = attach["Z"]
            else:
                # Fallback to the row's own coordinates if necessary.
                row["gridXNo"] = row.get("gridXNo", 0)
                row["gridZNo"] = row.get("gridZNo", 0)
                row["posX"] = row.get("posX", 0.0)
                row["posZ"] = row.get("posZ", 0.0)
                row["X"] = calc_world_x(int(row["gridXNo"]), float(row["posX"]))
                row["Z"] = calc_world_z(int(row["gridZNo"]), float(row["posZ"]))

            # Boss/location text fields are not present in the CSVs.
            # The JSON uses strings here, so we keep them blank unless we have a manual override.
            row["smallBaseMapName"] = ""
            row["smallBaseLocationName"] = ""
            row["smallBaseVariationName"] = ""

            # If the row looks like a real small-base row, you can optionally enrich it later.
            row_dicts.append(row)

        play_area1_id = to_int(p.get("playArea1"))
        play_area2_id = to_int(p.get("playArea2"))
        play_area1_coords = _resolve_play_area_coords(play_area1_id, small_lookup, attach_lookup)
        play_area2_coords = _resolve_play_area_coords(play_area2_id, small_lookup, attach_lookup)

        out = {
            "patternId": pattern_id,
            "rareMapIndex": RARE_MAP_INDEX,
            "rareMap": RARE_MAP_NAME,
            "bossId1": boss_id1,
            "bossId1Name": BOSS_ID_TO_NAME.get(boss_id1, ""),
            "bossId2": boss_id2,
            "bossId2Name": BOSS_ID_TO_NAME.get(boss_id2, ""),
            "extraBossId1": extra_boss_id1,
            "extraBossId1Name": BOSS_ID_TO_NAME.get(extra_boss_id1, "") if extra_boss_id1 != -1 else "",
            "extraBossId2": extra_boss_id2,
            "extraBossId2Name": BOSS_ID_TO_NAME.get(extra_boss_id2, "") if extra_boss_id2 != -1 else "",
            "playArea1": play_area1_id,
            "playArea2": play_area2_id,
            "playArea1_gridXNo": play_area1_coords["gridXNo"] if play_area1_coords else None,
            "playArea1_gridZNo": play_area1_coords["gridZNo"] if play_area1_coords else None,
            "playArea1_posX": play_area1_coords["posX"] if play_area1_coords else None,
            "playArea1_posZ": play_area1_coords["posZ"] if play_area1_coords else None,
            "playArea1_X": play_area1_coords["X"] if play_area1_coords else None,
            "playArea1_Z": play_area1_coords["Z"] if play_area1_coords else None,
            "playArea2_gridXNo": play_area2_coords["gridXNo"] if play_area2_coords else None,
            "playArea2_gridZNo": play_area2_coords["gridZNo"] if play_area2_coords else None,
            "playArea2_posX": play_area2_coords["posX"] if play_area2_coords else None,
            "playArea2_posZ": play_area2_coords["posZ"] if play_area2_coords else None,
            "playArea2_X": play_area2_coords["X"] if play_area2_coords else None,
            "playArea2_Z": play_area2_coords["Z"] if play_area2_coords else None,
            "rows": row_dicts,
        }
        output.append(out)

    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path, help="Path to LotResultPlayAreaParam.zip")
    ap.add_argument("-o", "--output", type=Path, default=Path("LotResultSmallBaseAndSpot_grouped_by_patternId_14_reconstructed.json"))
    args = ap.parse_args()

    data = reconstruct(args.zip_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}")
    print(f"Patterns: {len(data)}")
    unresolved = sum(
        1
        for p in data
        for key in (
            "playArea1_X", "playArea2_X"
        )
        if p.get(key) is None
    )
    if unresolved:
        print(
            "Note: some playArea coordinate fields were not resolved exactly "
            f"({unresolved} missing fields). See _resolve_play_area_coords()."
        )


if __name__ == "__main__":
    main()
