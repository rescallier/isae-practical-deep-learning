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
# %matplotlib notebook

# %%
import os
import sys

# %%
# add khumeia
sys.path.append("./src/")
sys.path = list(set(sys.path))

# %%
# setup env variable
os.environ['TP_DATA'] = "./data/"
raw_data_dir = os.path.join(os.environ.get("TP_DATA"), "raw")
TRAINVAL_DATA_DIR = os.path.join(raw_data_dir, "trainval")
EVAL_DATA_DIR = os.path.join(raw_data_dir, "eval")

# %% [markdown]
# ## Using Khumeia

# %%
from khumeia import helpers

# %%
trainval_dataset = helpers.dataset_generation.items_dataset_from_path(TRAINVAL_DATA_DIR)

# %% [markdown]
# ## Dataset parsing using khumeia

# %%
MAX_ITEMS = None
TILE_SIZE = 64
SPARSE_TILE_STRIDE = 64
DENSE_TILE_STRIDE = 16
MARGIN = 16

# %%
trainval_dataset.items = trainval_dataset.items[:min(len(trainval_dataset), MAX_ITEMS or len(trainval_dataset))]
train_dataset, test_dataset = helpers.dataset_generation.split_dataset(trainval_dataset, proportion=0.75)

# %%
from khumeia.roi.sliding_window import SlidingWindow

# %%
sliding_window_dense = SlidingWindow(tile_size=TILE_SIZE,
                                     stride=DENSE_TILE_STRIDE,
                                     margin_from_bounds=MARGIN,
                                     discard_background=True)
sliding_window_sparse = SlidingWindow(tile_size=TILE_SIZE,
                                      stride=SPARSE_TILE_STRIDE,
                                      margin_from_bounds=MARGIN,
                                      discard_background=False)

# %%
train_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    train_dataset, sliding_windows=[sliding_window_sparse, sliding_window_dense], n_jobs=4)
# %%
test_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    test_dataset, sliding_windows=[sliding_window_dense, sliding_window_sparse], n_jobs=4)

# %% [markdown]
# ## Toy Dataset Generation
# Let's generate our first dataset

# %%
NB_TRAIN_TILES = 6000
NB_TEST_TILE = 1500

# %%
from khumeia.roi.tiles_sampler import *
# %%
train_stratified_sampler = BackgroundToPositiveRatioSampler(
    nb_positive_tiles_max=NB_TRAIN_TILES,
    background_to_positive_ratio=1,
    with_replacement=True,
    shuffle=True,
)

train_tiles_sampled = helpers.dataset_generation.sample_tiles_from_candidates(train_tiles,
                                                                              tiles_samplers=[train_stratified_sampler])

# %%
test_stratified_sampler = BackgroundToPositiveRatioSampler(
    nb_positive_tiles_max=NB_TEST_TILE,
    background_to_positive_ratio=1,
    with_replacement=True,
    shuffle=True,
)

test_tiles_sampled = helpers.dataset_generation.sample_tiles_from_candidates(test_tiles,
                                                                             tiles_samplers=[test_stratified_sampler])

# %%
train_array = helpers.dataset_generation.dump_dataset_tiles(tiles_dataset=train_tiles_sampled,
                                                            items_dataset=train_dataset)

# %%
test_array = helpers.dataset_generation.dump_dataset_tiles(tiles_dataset=test_tiles_sampled, items_dataset=test_dataset)

# %%
train_images = np.asarray([i[0] for i in train_array.items])
train_labels = np.asarray([i[1] for i in train_array.items])

# %%
test_images = np.asarray([i[0] for i in test_array.items])
test_labels = np.asarray([i[1] for i in test_array.items])

# %%
print(test_images.shape)
print(test_labels.shape)

# %%
# Save as dict of nparrays
data_dir = os.environ.get("TP_DATA")
dataset_path = os.path.join(data_dir, "toy_aircraft_dataset.npz")

with open(dataset_path, "wb") as f:
    np.savez_compressed(f,
                        train_images=train_images,
                        train_labels=train_labels,
                        test_images=test_images,
                        test_labels=test_labels)

# %%
# upload to gcp
import subprocess
cmd = "gsutil -m cp -r {} gs://isae-deep-learning/".format(os.path.abspath(dataset_path))
print(cmd)
subprocess.check_call(cmd, shell=True)
# %%
# try to reload
ds = np.DataSource("/tmp/")
f = ds.open("https://storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz",'rb')
toy_dataset = np.load(f)
train_images = toy_dataset['train_images']
print(train_images.shape)
