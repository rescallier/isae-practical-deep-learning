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

# %% [markdown]
# # Sliding windows and samplers showcase

# %%
from khumeia.data.dataset import TilesDataset
dataset = TilesDataset(items=trainval_collection)

# %% [markdown]
# ## Multiple examples of sliding windows

# %% [markdown]
# ### Sliding with only one stride

# %%
from khumeia.data.dataset import SlidingWindow

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
# A demo with higher level functions in the framework
item = dataset.items[0]
tiles = dataset.candidate_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %%
# Change labelling mode from "center" to intersection.area / min(area1, area2)
sliding_window = SlidingWindow(
    tile_size=64,
    stride=64,
    discard_background=False,
    padding=0,
    label_assignment_mode="ioa",
    intersection_over_area_threshold=0.30)

dataset.generate_candidates_tiles(sliding_windows=sliding_window)

print(dataset)

# %%
# A demo with higher level functions in the framework
item = dataset.items[0]
tiles = dataset.candidate_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %% [markdown]
# ### Sliding windows with multiple strides

# %%
sliding_window = SlidingWindow(
    tile_size=64,
    stride=64,
    discard_background=False,
    padding=0,
    label_assignment_mode="center")

sliding_window_fine = SlidingWindow(
    tile_size=64,
    stride=16,
    discard_background=True,
    padding=0,
    label_assignment_mode="center")

dataset.generate_candidates_tiles(
    sliding_windows=[sliding_window, sliding_window_fine])

print(dataset)

# %%
# A demo with higher level functions in the framework
item = dataset.items[0]
tiles = dataset.candidate_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()

# %% [markdown]
# ## Samplers

# %%
from khumeia.data.sampler import *

# %%
# Random sampling
sampler = RandomSampler(nb_tiles_max=4000, with_replacement=False)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
sampler = RandomPerItemSampler(nb_tiles_max=4000, with_replacement=False)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# Manually sample n aircrafts and m background
sampler1 = RandomPerItemSampler(
    nb_tiles_max=1000, with_replacement=False, target_label="aircraft")
sampler2 = RandomPerItemSampler(
    nb_tiles_max=2000, with_replacement=False, target_label="background")
dataset.sample_tiles_from_candidates(tiles_samplers=[sampler1, sampler2])
print(dataset)

# %%
# Manually sample n aircrafts and m background with replacement
sampler1 = RandomPerItemSampler(
    nb_tiles_max=1000, with_replacement=True, target_label="aircraft")
sampler2 = RandomPerItemSampler(
    nb_tiles_max=2000, with_replacement=False, target_label="background")
dataset.sample_tiles_from_candidates(tiles_samplers=[sampler1, sampler2])
print(dataset)

# %%
# Advanced sampler: Stratification
sampler = StratifiedPerItemSampler(nb_tiles_max=4000, with_replacement=True)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# Advanced sampler: Background to foreground ratio
sampler = BackgroundToPositiveRatioPerItemSampler(
    background_to_positive_ratio=2,
    nb_positive_tiles_max=1000,
    with_replacement=True)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# Advanced sampler: Background to foreground ratio without replacement
sampler = BackgroundToPositiveRatioPerItemSampler(
    background_to_positive_ratio=2,
    nb_positive_tiles_max=None,
    with_replacement=False)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# Advanced sampler: Background to foreground ratio
sampler = BackgroundToPositiveRatioPerItemSampler(
    background_to_positive_ratio=2,
    nb_positive_tiles_max=1000,
    with_replacement=True)
dataset.sample_tiles_from_candidates(tiles_samplers=sampler)
print(dataset)

# %%
# A demo with higher level functions in the framework
item = dataset.items[0]
tiles = dataset.sampled_tiles

image = visualisation.draw_item_with_tiles(item, tiles)
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()
