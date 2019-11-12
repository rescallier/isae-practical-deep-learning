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
# # jupyter magic commands in order to set everything up.
# %autosave 0
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook

# %%
# Install the package
# %pip install git+https://github.com/fchouteau/isae-practical-deep-learning.git#egg=khumeia\&subdirectory=src

# %%
# # Setup our environment variable needed to locate data

# # You should change it to your dev env value (likely ./data)
# # %env TP_DATA = ../data/

import os
os.environ['TP_DATA'] = "../data/"

# %% [markdown]
# # Very Basic Exploratory Data Analysis
# You should always start by taking a look at the data. This notebook will walk you through the steps of downloading the training dataset, opening the images, counting labels and actually visualising images with the aircrafts and images' histograms
# Feel free to add more visualisation and data exploration to this example.

# %%
# Global imports
import os
import pandas as pd
import numpy as np
import scipy.stats
import tqdm

# %%
# let's first download our training data
import khumeia

khumeia.helpers.download_train_data()

# %%
# Let's explore what we downloaded
for root, dirs, files in os.walk(os.path.join(os.environ.get("TP_DATA"), "raw")):
    for file in files:
        print(os.path.join(root, file))

# %% [markdown]
# ## Use the pandas dataframe to get a quick look at our data

# %%
raw_data_dir = os.path.join(os.environ.get("TP_DATA"), "raw")

# %%
image_ids = pd.read_csv(os.path.join(raw_data_dir, "trainval_ids.csv"))
train_labels = pd.read_csv(os.path.join(raw_data_dir, "trainval_labels.csv"))

# %%
print("Number of images in train dataset {}".format(train_labels['image_id'].value_counts()))

# %%
print("Description of labels \n{}".format(scipy.stats.describe(train_labels['image_id'].value_counts())))

# %%
# Size of different objects
train_labels['size'].describe()

# %% [markdown]
# ## Loading data using the khumeia framework

# %%
from khumeia import helpers

# %%
TRAINVAL_DATA_DIR = os.path.join(raw_data_dir, "trainval")

# %%
trainval_dataset = helpers.dataset_generation.items_dataset_from_path(TRAINVAL_DATA_DIR)

# %%
for satellite_image in trainval_dataset:
    print(satellite_image)

# %% [markdown]
# ## Plotting histograms and descriptions

# %%
# Let's write a histogram function
from matplotlib import pyplot as plt


def plot_histogram(dataset, n_bins=256):
    """
    Plotting histogram over a dataset
    Args:
        dataset(khumeia.data.Dataset): dataset
        n_bins(int): number of bins for histogram

    Returns:
        The histogram
    """
    mean_hist_r = [0 for _ in range(n_bins)]
    mean_hist_g = [0 for _ in range(n_bins)]
    mean_hist_b = [0 for _ in range(n_bins)]

    for image_item in tqdm.tqdm(dataset, desc='computing histograms...'):
        img = image_item.image

        hist_r, _ = np.histogram(img[:, :, 0], bins=n_bins, density=True)
        hist_g, _ = np.histogram(img[:, :, 1], bins=n_bins, density=True)
        hist_b, _ = np.histogram(img[:, :, 2], bins=n_bins, density=True)
        mean_hist_r = np.sum([mean_hist_r, hist_r], axis=0)
        mean_hist_g = np.sum([mean_hist_g, hist_g], axis=0)
        mean_hist_b = np.sum([mean_hist_b, hist_b], axis=0)

    mean_hist_r /= len(image_ids)
    mean_hist_g /= len(image_ids)
    mean_hist_b /= len(image_ids)

    plt.bar(np.arange(len(mean_hist_r)), mean_hist_r, color='red', width=1, alpha=0.5)
    plt.bar(np.arange(len(mean_hist_g)), mean_hist_g, color='green', width=1, alpha=0.5)
    plt.bar(np.arange(len(mean_hist_b)), mean_hist_b, color='blue', width=1, alpha=0.5)
    plt.show()


# %%
# Plot the histogram for the 10 images
plot_histogram(trainval_dataset[:10])


# %%
def describe_dataset(dataset):
    """
    Print image id and image shape and nb of labels per item
    Args:
        dataset:

    Returns:

    """
    for image_item in dataset:
        print("{} - {} - {}".format(image_item.image_id, image_item.shape, len(image_item.labels)))


# %%
describe_dataset(trainval_dataset)

# %% [markdown]
# ## Some data visualisation
# Let's plot an image (using khumeia helpers) and its labels

# %%
item = trainval_dataset[2]
print(item)
image = item.image
labels = item.labels

# %%
image = helpers.visualisation.draw_bboxes_on_image(image, labels, color="green")
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()
