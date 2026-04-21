
import argparse, json
from layout_classifier_lib import classify_image, configure_logging, load_pickle

p = argparse.ArgumentParser(description='Infer Mountaintop layout from one image')
p.add_argument('--model', required=True)
p.add_argument('--input', required=True)
p.add_argument('--metadata', help='Accepted for CLI compatibility; model already contains metadata.', default=None)
p.add_argument('--verbose', action='store_true')
args = p.parse_args()
configure_logging(args.verbose)
model = load_pickle(args.model)
out = classify_image(args.input, model)
print(json.dumps(out, indent=2))
