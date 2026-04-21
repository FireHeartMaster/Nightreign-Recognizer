import json, pathlib
root = pathlib.Path(r'C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer')
pattern_dir = root / 'great_hollow_pattern_sets'
field_counts = {}
for p in pattern_dir.glob('pattern_*.json'):
    data = json.loads(p.read_text(encoding='utf-8'))
    for item in data.get('activeMapPatternSetFields', []):
        field_counts[item.get('field', '')] = field_counts.get(item.get('field', ''), 0) + 1
print('unique fields', len(field_counts))
for field, count in sorted(field_counts.items(), key=lambda x: (-x[1], x[0]))[:50]:
    print(count, field)
