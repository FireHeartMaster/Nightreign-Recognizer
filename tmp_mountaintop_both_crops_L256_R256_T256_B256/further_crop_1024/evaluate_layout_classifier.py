
import argparse, json
from layout_classifier_lib import configure_logging, evaluate_images, load_pickle

p = argparse.ArgumentParser(description='Evaluate Mountaintop layout classifier')
p.add_argument('--images', required=True)
p.add_argument('--metadata', required=False, default=None)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--verbose', action='store_true')
args = p.parse_args()
configure_logging(args.verbose)
model = load_pickle(args.model)
report = evaluate_images(args.images, model)
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(json.dumps({'output': args.output, 'top1_accuracy': report['top1_accuracy'], 'top3_accuracy': report['top3_accuracy'], 'top5_accuracy': report['top5_accuracy']}, indent=2))
