#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from layout_classifier_lib import build_query_splits, configure_logging, evaluate_gallery_images, evaluate_queries, load_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Noklateo layout matcher model')
    parser.add_argument('--images', required=True, help='Directory containing further_crop_1024 PNG images')
    parser.add_argument('--model', required=True, help='Path to trained model pickle')
    parser.add_argument('--metadata', required=False, help='Accepted for CLI compatibility; metadata is embedded in the model')
    parser.add_argument('--output', required=True, help='Output report JSON path')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    configure_logging(args.verbose)
    model = load_pickle(args.model)
    split_queries = build_query_splits(args.images, args.seed)
    report = {
        'original_gallery_images': evaluate_gallery_images(args.images, model),
        'synthetic_query_splits': {name: evaluate_queries(samples, model) for name, samples in split_queries.items()},
        'calibration': model.get('calibration', {}),
        'model_config': model.get('config', {}),
        'metadata_diagnostics': model.get('metadata_diagnostics', {}),
        'duplicate_report': model.get('duplicate_report', {}),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved evaluation report to {out}')


if __name__ == '__main__':
    main()
