#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from layout_classifier_lib import build_gallery, configure_logging, evaluate_on_queries, image_to_gray, list_images, read_image, save_pickle, tune_hyperparameters

def main():
    parser = argparse.ArgumentParser(description='Train Limveld layout classifier')
    parser.add_argument('--images', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', default='models/layout_model.pkl')
    parser.add_argument('--report', default='models/training_report.json')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    configure_logging(args.verbose)
    model = build_gallery(args.images, args.metadata)
    calibration = tune_hyperparameters(model, args.images, seed=args.seed)
    model['calibration'] = calibration
    model['train_args'] = vars(args)
    records = list_images(args.images)
    original_queries = []
    split_queries = {'train': [], 'val': [], 'test': []}
    split_map = calibration['synthetic_split']
    for rec in records:
        gray = image_to_gray(read_image(rec.path))
        original_queries.append((rec.pattern_id, Path(rec.path).name, gray))
        bucket = 'train' if rec.pattern_id in split_map['train'] else ('val' if rec.pattern_id in split_map['val'] else 'test')
        split_queries[bucket].append((rec.pattern_id, Path(rec.path).name, gray))
    report = {'model_summary': {'num_patterns': int(len(model['pattern_ids'])), 'num_images': int(len(model['image_paths'])), 'metadata_diagnostics': model['metadata_diagnostics'], 'calibration': calibration}, 'split_reports': {k: evaluate_on_queries(v, model) for k, v in split_queries.items()}, 'original_gallery_report': evaluate_on_queries(original_queries, model), 'limitations': ['Dataset has one canonical image per patternId, so this is a closed-set gallery matcher rather than a conventional classifier.', 'Train/val/test reports are deterministic patternId partitions of the gallery because the dataset has one canonical image per class.']}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_pickle(model, args.output)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'saved_model': args.output, 'saved_report': args.report, 'num_patterns': len(model['pattern_ids']), 'original_top1': report['original_gallery_report']['top1_accuracy']}, indent=2))

if __name__ == '__main__':
    main()
