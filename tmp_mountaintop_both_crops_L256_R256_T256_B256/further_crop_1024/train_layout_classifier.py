
import argparse, json
from layout_classifier_lib import build_gallery, calibrate_model, configure_logging, save_pickle

p = argparse.ArgumentParser(description='Train Mountaintop layout classifier')
p.add_argument('--images', required=True)
p.add_argument('--metadata', required=True)
p.add_argument('--output', '--output-model', dest='output', required=True)
p.add_argument('--report', '--output-report', dest='report', default='models/training_report.json')
p.add_argument('--seed', type=int, default=1337)
p.add_argument('--verbose', action='store_true')
args = p.parse_args()
configure_logging(args.verbose)
model = build_gallery(args.images, args.metadata)
model['calibration'] = calibrate_model(model, args.images, args.seed)
save_pickle(model, args.output)
report = {'task':'Mountaintop layout patternId classification from one image','model_config':model['config'],'metadata_diagnostics':model['metadata_diagnostics'],'calibration':model['calibration'],'gallery_count': len(model['pattern_ids'])}
with open(args.report, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(json.dumps({'model_path': args.output, 'report_path': args.report, 'gallery_count': len(model['pattern_ids'])}, indent=2))
