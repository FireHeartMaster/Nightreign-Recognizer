#!/usr/bin/env python3
import argparse, json
from layout_classifier_lib import classify_image, configure_logging, load_pickle

def main():
    parser = argparse.ArgumentParser(description='Infer Limveld layout patternId from one image')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--metadata', required=False, help='Accepted for CLI compatibility; metadata is embedded in the model')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    configure_logging(args.verbose)
    model = load_pickle(args.model)
    print(json.dumps(classify_image(args.input, model), indent=2))

if __name__ == '__main__':
    main()
