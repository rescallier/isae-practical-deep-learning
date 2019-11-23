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

# %% [markdown]
# # Session 2 part 1: Discovering skorch, class-imbalance

# %%
# Put your imports here
import numpy as np

# %%
# Global variables
trainval_dataset_url = "https://storage.googleapis.com/isae-deep-learning/trainval_aircraft_dataset.npz"

# %% {"tags": ["exercise"]}
# This cell should not be exported

# %% [markdown]
# ## Downloading the dataset

# %%
ds = np.DataSource("/tmp/")
f = ds.open(trainval_dataset_url, 'rb')
trainval_dataset = np.load(f)
train_images = trainval_dataset['train_images']
train_labels = trainval_dataset['train_labels']
test_images = trainval_dataset['test_images']
test_labels = trainval_dataset['test_labels']
