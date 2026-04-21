#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from layout_classifier_lib import (
    build_gallery,
    build_query_splits,
    calibrate_model,
    configure_logging,
    evaluate_gallery_images,
    evaluate_queries,
    parse_args_common,
    save_pickle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train/build The Crater layout matcher model')
    parser.add_argument('--images', required=True, help='Directory containing further_crop_1024 PNG images')
    parser.add_argument('--output-model', default='models/layout_model.pkl', help='Output model pickle path')
    parser.add_argument('--output-report', default='models/training_report.json', help='Output training report JSON path')
    parser.add_argument('--seed', type=int, default=1337)
    parse_args_common(parser)
    args = parser.parse_args()

    configure_logging(args.verbose)
    model = build_gallery(args.images, args.metadata)
    model['calibration'] = calibrate_model(model, args.images, args.seed)

    split_queries = build_query_splits(args.images, args.seed)
    split_reports = {name: evaluate_queries(samples, model) for name, samples in split_queries.items()}
    original_gallery_report = evaluate_gallery_images(args.images, model)

    report = {
        'task': 'The Crater layout patternId classification from one image',
        'protocol_note': model['calibration']['split_protocol'],
        'model_config': model['config'],
        'metadata_diagnostics': model['metadata_diagnostics'],
        'duplicate_report': model['duplicate_report'],
        'calibration': model['calibration'],
        'evaluation': {
            'original_gallery_images': original_gallery_report,
            'synthetic_query_splits': split_reports,
        },
    }

    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(model, str(out_model))

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved model to {out_model}')
    print(f'Saved report to {out_report}')


if __name__ == '__main__':
    main()
