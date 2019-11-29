# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Session 2 Part 1: Going Further, Discovering class-imbalance in datasets

# %%
# Put your imports here
import numpy as np

# %%
# Global variables
TRAINVAL_DATASET_URL = "https://storage.googleapis.com/isae-deep-learning/trainval_aircraft_dataset.npz"

# %% {"tags": ["exercise"]}
# This cell should not be exported

# %% [markdown]
# ## Downloading the dataset

# %%
ds = np.DataSource("/tmp/")
f = ds.open(TRAINVAL_DATASET_URL, 'rb')
trainval_dataset = np.load(f)
train_images = trainval_dataset['train_images']
train_labels = trainval_dataset['train_labels']
test_images = trainval_dataset['test_images']
test_labels = trainval_dataset['test_labels']


# %% [markdown]
# ## Q1. During Session 1, you learnt how to set up your environment on GCP, train a basic CNN on a small training set and plot metrics. Now let's do it again !
#
# Once you have downloaded data, use the notebook from Session 1 to get:
#
# a. Visualisation of the data
#
# b. Training of the model using steps seen during Session 1
#
# c. Metrics based on this training
#
# d. Comparison of metrics between this new dataset and the one from Session 1
#
# e. What did you expect ?

# %%
# Q1


# %% [markdown]
# ## If you need to take a step back, the recipe to success is http://karpathy.github.io/2019/04/25/recipe/ 
#
# ![image.png](attachment:image.png)

# %% [markdown]
# ## Q2. Let's improve our model's performance
#
# ![image.png](slides/static/img/mlsystem.png)

# %% [markdown]
# ### a. Solving the imbalanced data problem
#
# Go through your data: is the dataset balanced ? If now, which steps can I do to solve this imbalance problem ?
#
# If you need help on this step, refer [to this tutorial on how to tackle imbalanced dataset](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)
#
# - Which step would you take ?
# - Don't forget to apply the same step on you train and validation dataset
#
# Try to decide and a method to modify only the dataset and rerun your training. Did performance improve ?

# %%
# Q2.a here

# %% [markdown]
# ### b. Optimizer and model modifications
#
# i ) Now that you have worked on your dataset and decided to undersample it, it's time to tune your network and your training configuration
#
# In Session 1, you tested two descent gradient. What is the effect of its modification? Apply it to your training and compare metrics.
#
# ii ) An other important parameter is the learning rate, you can [check its effect on the behavior of your training](https://developers.google.com/machine-learning/crash-course/fitter/graph).
#
# iii) There is no absolute law concerning the structure of your deep Learning model. During the [Deep Learning class](https://github.com/erachelson/MLclass/blob/master/7%20-%20Deep%20Learning/Deep%20Learning.ipynb) you had an overview of existing models 
#
# You can operate a modification on your structure and observe the effect on final metrics. Of course, remain consistent with credible models, cf Layer Patterns chapter on this "must view" course : http://cs231n.github.io/convolutional-networks/
#
# ![image.png](slides/static/img/comparison_architectures.png)

# %%
# Q2.b here

# %% [markdown]
# ### c. Going Further
#
# Here is an overview of [possible hyperparameter tuning when training Convolutional Neural Networks](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8)
#
# You can try and apply those techniques to your use case.
#
# - Does these techniques yield good results ? What about the effort-spent-for-performance ratio ?
# - Do you find it easy to keep track of your experiments ? 
# - What would you need to have a better overview of the effects of these search ?
#
# Don't spend too much time on this part as the next is more important. You can come back to it after you're finished

# %%
# Q2.c here

# %% [markdown]
# ## Q3. Test and Improve
#
# a. Now that you have optimised your structure for a given problem, test it on the whole initial dataset to see its metrics.
#
# b. Check the output of your model. The best is to learn from your failure: select and save pictures of when your model provided bad outputs, and retrain your network using those data. 
# How did the final metrics evolve ?
#
# c .  !!!! SAVE YOUR MODEL !!! 
#
# ## Did you save your model ??
