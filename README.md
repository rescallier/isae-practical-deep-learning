# Deep Learning for Computer Vision, practical session at ISAE-SUPAERO

## Introduction

This repository contains the code and documentation of a "Deep Learning practical session" given at ISAE-SUPAERO on November 13th 2018.

More information, guidelines and API documentation are available in the [documentation website](https://fchouteau.github.io/isae-practical-deep-learning/site/index.html). It is recommended to read it first as it contains the necessary information to run this from scratch.

You can find a rough guideline and usage references in the jupyter notebooks in `jupyter/`

## Prerequisites

TODO

## Where to run it ?

This hands on session is based on running code & training using [Google Cloud Platform Deep Learning VMs](https://cloud.google.com/deep-learning-vm/), see `gcp/` for examples on configuring your own machine. However, this is theoretically runnable everywhere since even data access in managed by the included python package.

Should you want to do this at home you can use Google Collaboratory instances: https://colab.research.google.com/

## Installation

Just install the framework (it was design on python 3.5+ so support is not guaranteed on python 2.7) with pip: 

`pip install git+https://github.com/fchouteau/isae-practical-deep-learning.git#egg=khumeia\&subdirectory=src`

The `khumeia` framework aims at facilitating interaction with data before and after training your models, as well as mocking up larger scale inference on full images. It is based on remote sensing imagery and tiles classification.

## Usage & Contribution

No support is guaranteed by the authors beyond the hands-on session.

This hands-on session was created by Florient Chouteau and Matthieu Le Goff.

See [`licence.md`](./licence.md) for licence information.