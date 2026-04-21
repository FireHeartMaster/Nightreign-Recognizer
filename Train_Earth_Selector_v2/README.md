# Shifting Earth Selector v2

Train:
python train_earth_selector_v2.py --sources limveld_10_further_crops_with_annotations_txt.zip mountaintop_10_further_crops_with_annotations_txt.zip the_crater_10_further_crops_with_annotations_txt.zip rotted_woods_10_further_crops_with_annotations_txt.zip noklateo_10_further_crops_with_annotations_txt.zip great_hollow_10_further_crops_with_annotations_txt.zip --model earth_selector_v2_model.pkl

Infer single image:
python infer_earth_selector_v2.py --model earth_selector_v2_model.pkl --input some_further_crop.png

Infer folder:
python infer_earth_selector_v2.py --model earth_selector_v2_model.pkl --input ./folder_with_images --output-json earth_predictions.json --output-csv earth_predictions.csv
