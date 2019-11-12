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

# %% [markdown]
# # Generating training datasets

# %% [markdown]
# ## Reload Data

# %%
import json
import os

# %%
from khumeia.data.item import SatelliteImage

# %%
RAW_DATA_DIR = os.path.join(os.environ.get("TP_ISAE_DATA"),"raw")
TRAINVAL_DATA_DIR = os.path.join(RAW_DATA_DIR, "trainval")

# %%
trainval_collection = SatelliteImage.list_items_from_path(TRAINVAL_DATA_DIR)

# We reduce the collection to 2 items for this demo in order to run cells faster
trainval_collection = trainval_collection[:2]

# %%
print(trainval_collection)

# %%
from khumeia.data.dataset import TilesDataset, SlidingWindow
from khumeia.data.sampler import *

# %%
dataset = TilesDataset(items=trainval_collection)

# %% [markdown]
# ## Apply a sliding window

# %%
sliding_window = SlidingWindow(
    tile_size=64,
    stride=64,
    discard_background=False,
    padding=0,
    label_assignment_mode="center")

dataset.generate_candidates_tiles(sliding_windows=sliding_window)

print(dataset)

# %%
# What does it look like ?
# %matplotlib notebook

from khumeia.utils import list_utils
from khumeia import visualisation
from matplotlib import pyplot as plt

# %%
item = dataset.items[0]
image = item.image
labels = item.labels

tiles = list_utils.filter_tiles_by_item(dataset.candidate_tiles, item)
print(len(tiles))

aircrafts_tiles = list_utils.filter_tiles_by_label(tiles, "aircraft")
print(len(aircrafts_tiles))

background_tiles = list_utils.filter_tiles_by_label(tiles, "background")
print(len(background_tiles))

# %%
image = visualisation.draw_bboxes_on_image(image, background_tiles, color=(255,0,0))
image = visualisation.draw_bboxes_on_image(image, aircrafts_tiles, color=(0,0,255))
image = visualisation.draw_bboxes_on_image(image, labels, color=(0,255,0))

plt.figure(figsize=(10,10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %%
# A demo with higher level functions in the framework
item = dataset.items[1]
tiles = dataset.candidate_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10,10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %% [markdown]
# ## Apply a sampler to select tiles among candidates

# %%
# Random sampling of 4000 tiles from our dataset
sampler = RandomSampler(nb_tiles_max=4000, with_replacement=False)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# A demo with higher level functions in the framework
item = dataset.items[1]
tiles = dataset.sampled_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10,10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %% [markdown]
# ## Dumping data
#
# Once you have selected the correct sampling methodology,

# %%
# Now dump data to keras.ImageDataGenerator format

dataset_dir = os.path.join(os.path.expandvars("$TP_ISAE_DATA"), "dataset")

## Uncomment to dump
# dataset.generate_tiles_dataset(output_dir=dataset_dir,remove_first=True)
