# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib notebook
# %config Application.log_level="INFO"

# %%
# %env TP_ISAE_DATA = /home/fchouteau/repositories/tp_isae/data/

# %%
# Load data
import json
import os
from khumeia.data.item import SatelliteImage

RAW_DATA_DIR = os.path.join(os.environ.get("TP_ISAE_DATA"), "raw")
TRAINVAL_DATA_DIR = os.path.join(RAW_DATA_DIR, "trainval")

trainval_collection = SatelliteImage.list_items_from_path(TRAINVAL_DATA_DIR)

# We reduce the collection to 2 items for this demo in order to run cells faster
trainval_collection = trainval_collection[:2]

print(trainval_collection)

# %%
# Import vis
# %matplotlib notebook

from khumeia.utils import list_utils
from khumeia import visualisation
from matplotlib import pyplot as plt

#%%

# Load data
import json
import os

import khumeia
khumeia.download_eval_data()

#%%

from khumeia.data.item import SatelliteImage

RAW_DATA_DIR = os.path.join(os.environ.get("TP_ISAE_DATA"), "raw")
EVAL_DATA_DIR = os.path.join(RAW_DATA_DIR, "eval")

eval_collection = SatelliteImage.list_items_from_path(EVAL_DATA_DIR)

print(eval_collection)

#%%

from khumeia.inference.engine import InferenceEngine
from khumeia.data.sliding_window import SlidingWindow
from khumeia.inference.predictor import Predictor

#%%

import random
import numpy as np


class DemoPredictor(Predictor):
    """
    Dummy predictor randomly returning aircraft or background
    """
    def __init__(self, threshold=0.9, batch_size=128):
        self.threshold = threshold
        self.batch_size = batch_size
        self.model = lambda x: "aircraft" if random.random() > threshold else "background"

    def predict(self, tile_data):
        print("Received data of shape {}".format(tile_data.shape))
        return self.model(tile_data)

    def predict_on_batch(self, tiles_data):
        print(len(tiles_data))
        tiles_data = np.asarray(tiles_data)
        print("Received data of shape {}".format(tiles_data.shape))
        return [self.model(tile_data) for tile_data in tiles_data]


#%%

predictor = DemoPredictor(threshold=0.75, batch_size=128)

#%%

sliding_window = SlidingWindow(tile_size=64,
                               stride=64,
                               discard_background=False,
                               padding=0,
                               label_assignment_mode="center")

#%%

inference_engine = InferenceEngine(items=eval_collection)

#%%

results = inference_engine.predict_on_item(eval_collection[0], predictor=predictor, sliding_windows=sliding_window)

#%%

item = eval_collection[0]
image = item.image
labels = item.labels

tiles = list(filter(lambda tile: tile.item_id == item.key, results))
true_positives = list(filter(lambda tile: tile.is_true_positive, tiles))
false_positives = list(filter(lambda tile: tile.is_false_positive, tiles))
false_negatives = list(filter(lambda tile: tile.is_false_negative, tiles))

image = visualisation.draw_bboxes_on_image(image, labels, color=(255, 255, 255))
image = visualisation.draw_bboxes_on_image(image, true_positives, color=(0, 255, 0))
image = visualisation.draw_bboxes_on_image(image, false_positives, color=(0, 0, 255))
image = visualisation.draw_bboxes_on_image(image, false_negatives, color=(255, 0, 0))

plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

#%%
