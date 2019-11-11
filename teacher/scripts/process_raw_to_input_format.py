import os
import json
import glob
import pandas as pd

raw_data_dir = os.path.join(os.environ.get("TP_ISAE_DATA"), "raw")


def make_labels(fold="trainval"):
    df = pd.read_csv(os.path.join(raw_data_dir, '{}_ids.csv'.format(fold)))
    trainval_labels = []
    eval_labels = []
    image_ids = list(df['image_id'].unique())

    for image_id in image_ids:
        fold = list(df[df['image_id'] == image_id]['fold'])[0]
        image_path = os.path.join(raw_data_dir, fold, image_id + ".jpg")
        label_path = image_path.replace('.jpg', '.json')
        with open(label_path, "r") as f:
            labels = json.load(f)
        image_id = os.path.splitext(os.path.basename(label_path))[0]

        for label in labels['markers']:
            x, y, w = label['x'], label['y'], label['w']
            if fold == 'trainval':
                trainval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})
            else:
                eval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})

    pd.DataFrame(trainval_labels).to_csv(
        os.path.join(raw_data_dir, "{}_labels.csv".format(fold)), index=None, index_label=None)


def make_image_ids():
    list_train_images = glob.glob(os.path.join(raw_data_dir, "trainval", "*.jpg"), recursive=True)
    dataset = []
    for image_file in list_train_images:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        dataset.append({'image_id': image_id, 'fold': 'trainval'})
    pd.DataFrame(dataset).to_csv(os.path.join(raw_data_dir, "eval_ids.csv"), index_label=None, index=None)

    list_eval_images = glob.glob(os.path.join(raw_data_dir, "eval", "*.jpg"), recursive=True)
    dataset = []
    for image_file in list_eval_images:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        dataset.append({'image_id': image_id, 'fold': 'eval'})
    pd.DataFrame(dataset).to_csv(os.path.join(raw_data_dir, "trainval_ids.csv"), index_label=None, index=None)


make_image_ids()
make_labels("trainval")
make_labels("eval")
