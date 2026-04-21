#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from layout_classifier_lib import classify_image, configure_logging, list_images, load_pickle

def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = max(0, min(len(s)-1, int(round((len(s)-1) * q))))
    return s[idx]

def main():
    parser = argparse.ArgumentParser(description='Evaluate Limveld layout classifier')
    parser.add_argument('--images', required=True)
    parser.add_argument('--metadata', required=False, help='Accepted for CLI compatibility; metadata is embedded in the model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    configure_logging(args.verbose)
    model = load_pickle(args.model)
    records = list_images(args.images)
    rows = []
    y_true, rankings, margins, latencies, decided = [], [], [], [], []
    labels = [int(x) for x in model['pattern_ids']]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for rec in records:
        result = classify_image(rec.path, model)
        y_true.append(rec.pattern_id)
        rankings.append([item['patternId'] for item in result['top5']])
        margins.append(result['margin'])
        latencies.append(result['latency_ms'])
        decided.append(not result['unknown'])
        cm[label_to_index[rec.pattern_id]][label_to_index[result['best_patternId']]] += 1
        rows.append({'file': Path(rec.path).name, 'truth': rec.pattern_id, 'best_patternId': result['best_patternId'], 'unknown': result['unknown'], 'margin': result['margin'], 'confidence': result['confidence'], 'latency_ms': result['latency_ms'], 'top5': result['top5']})
    def topk(k):
        return sum(1 for truth, ranked in zip(y_true, rankings) if truth in ranked[:k]) / max(len(y_true), 1)
    decided_rows = [row for row in rows if not row['unknown']]
    decided_acc = (sum(1 for row in decided_rows if row['truth'] == row['best_patternId']) / len(decided_rows)) if decided_rows else None
    report = {'model_path': args.model, 'num_images': len(rows), 'top1_accuracy': topk(1), 'top3_accuracy': topk(3), 'top5_accuracy': topk(5), 'unknown_rate': 1.0 - (sum(decided) / max(len(decided), 1)), 'accuracy_on_decided_samples': decided_acc, 'margin_distribution': {'min': min(margins) if margins else None, 'p25': pct(margins, 0.25), 'median': pct(margins, 0.5), 'p75': pct(margins, 0.75), 'max': max(margins) if margins else None}, 'latency_ms': {'mean': sum(latencies) / len(latencies) if latencies else None, 'median': pct(latencies, 0.5), 'p95': pct(latencies, 0.95), 'max': max(latencies) if latencies else None}, 'confusion_matrix': {'labels': labels, 'matrix': cm}, 'rows': rows, 'model_metadata_diagnostics': model.get('metadata_diagnostics', {}), 'calibration': model.get('calibration', {})}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ['num_images', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy', 'unknown_rate', 'accuracy_on_decided_samples']}, indent=2))

if __name__ == '__main__':
    main()
