#!/usr/bin/env python3
import argparse
import json

from layout_classifier_lib import classify_image, configure_logging, load_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description='Infer The Crater layout patternId from one image')
    parser.add_argument('--model', required=True, help='Path to trained model pickle')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--metadata', required=False, help='Accepted for CLI compatibility; metadata is embedded in the model')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    configure_logging(args.verbose)
    model = load_pickle(args.model)
    result = classify_image(args.input, model)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
