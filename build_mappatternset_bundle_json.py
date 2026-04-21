
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_numeric_string(value):
    if value is None:
        return False
    s = str(value).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build one JSON file per patternId linking LotResultMapPatternFlag bundles to MapPatternSet rows."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output-dir", required=True, help="Directory to write JSON outputs")
    parser.add_argument("--pattern-id", type=int, default=None, help="Optional single patternId to export")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    play_area_path = input_dir / "LotResultPlayAreaParam.csv"
    flag_path = input_dir / "LotResultMapPatternFlag.csv"
    mappatternset_path = input_dir / "MapPatternSet.csv"

    play_area_rows = read_csv_rows(play_area_path)
    flag_rows = read_csv_rows(flag_path)
    mappattern_rows = read_csv_rows(mappatternset_path)

    play_area_by_id = {}
    for row in play_area_rows:
        row_id = parse_int(row.get("ID"))
        if row_id is not None:
            play_area_by_id[row_id] = row

    flags_by_pattern_id = defaultdict(list)
    for row in flag_rows:
        pattern_id = parse_int(row.get("patternId"))
        if pattern_id is not None:
            flags_by_pattern_id[pattern_id].append(row)

    mappatternset_by_id = {}
    for row in mappattern_rows:
        row_id = parse_int(row.get("ID"))
        if row_id is not None:
            mappatternset_by_id[row_id] = row

    pattern_ids = sorted(play_area_by_id.keys())
    if args.pattern_id is not None:
        pattern_ids = [pid for pid in pattern_ids if pid == args.pattern_id]

    created = 0
    skipped = 0
    errors = []

    for pattern_id in pattern_ids:
        play_area_row = play_area_by_id.get(pattern_id)
        if play_area_row is None:
            skipped += 1
            continue

        flag_group = flags_by_pattern_id.get(pattern_id, [])
        if not flag_group:
            skipped += 1
            continue

        pattern_set_ids = []
        for row in flag_group:
            psid = parse_int(row.get("patternSetId"))
            if psid is not None:
                pattern_set_ids.append(psid)

        unique_pattern_set_ids = sorted(set(pattern_set_ids))
        if len(unique_pattern_set_ids) != 1:
            errors.append(
                f"patternId {pattern_id} has inconsistent patternSetId values: {unique_pattern_set_ids}"
            )
            continue

        pattern_set_id = unique_pattern_set_ids[0]
        matched_mappattern_row = mappatternset_by_id.get(pattern_set_id)
        if matched_mappattern_row is None:
            errors.append(
                f"patternId {pattern_id} refers to missing MapPatternSet.ID {pattern_set_id}"
            )
            continue

        realized_modifier_bundle_ordered = []
        for row in flag_group:
            modifier = parse_int(row.get("modifier"))
            if modifier is not None and modifier != 0:
                realized_modifier_bundle_ordered.append(modifier)

        realized_modifier_bundle_unique_sorted = sorted(set(realized_modifier_bundle_ordered))
        realized_modifier_lookup = set(realized_modifier_bundle_ordered)

        active_fields = []
        for field_name, raw_value in matched_mappattern_row.items():
            if field_name == "ID":
                continue
            if not is_numeric_string(raw_value):
                continue
            numeric_value = parse_int(raw_value)
            if numeric_value is None:
                continue
            if numeric_value in realized_modifier_lookup:
                active_fields.append({
                    "field": field_name,
                    "value": numeric_value,
                })

        boss_modifiers = {
            "bossModifier1": parse_int(play_area_row.get("bossModifier1")),
            "bossModifier2": parse_int(play_area_row.get("bossModifier2")),
        }

        payload = {
            "patternId": pattern_id,
            "patternSetId": pattern_set_id,
            "bossModifiers": boss_modifiers,
            "realizedModifierBundleOrdered": realized_modifier_bundle_ordered,
            "realizedModifierBundleUniqueSorted": realized_modifier_bundle_unique_sorted,
            "activeMapPatternSetFields": active_fields,
            "matchedMapPatternSetRow": matched_mappattern_row,
            "sourceLotResultMapPatternFlagRows": flag_group,
            "sourceLotResultPlayAreaParamRow": play_area_row,
        }

        out_path = output_dir / f"pattern_{pattern_id:04d}_mappatternset_bundle.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        created += 1

    print(json.dumps({
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
