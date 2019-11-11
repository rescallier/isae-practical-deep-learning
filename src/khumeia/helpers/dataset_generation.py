"""
Generating dataset helpers
"""
import glob
import os
import shutil
import random

import numpy as np

from khumeia.data.item import SatelliteImage
from khumeia import LOGGER
from khumeia.data.dataset import Dataset
from khumeia.roi.sliding_window import SlidingWindow
from khumeia.roi.tiles_dumper import ItemTileDumper
from khumeia.roi.tiles_sampler import TilesSampler
from khumeia.utils import roi_list_utils

random.seed(2019)


def items_dataset_from_path(path=None):
    """
    Get a list of Satellite Images items from path

    Args:
        path: folder where to look

    Returns:
        list(SatelliteImageItem):
    """
    assert path is not None, "Please set folder variable, likely ${TP_DATA}/raw/trainval/"

    LOGGER.info("Looking in {}".format(path))
    items = []
    list_images = glob.glob(os.path.join(path, "*.jpg"))

    for image_file in list_images:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        item = SatelliteImage.from_image_id_and_path(image_id, path)
        # Read the when initialising to put data into cache
        LOGGER.info("Found item {}".format(item.image_id))
        assert isinstance(item.image, np.ndarray)
        assert isinstance(item.labels, list)
        items.append(item)

    items = list(sorted(items, key=lambda item: item.key))
    LOGGER.info("Found {} items".format(len(items)))

    return Dataset(items=items)


def split_dataset(dataset, proportion=0.75, shuffle=True):
    """
    Split dataset
    Args:
        dataset:
        proportion:
        shuffle:

    Returns:

    """
    if shuffle:
        random.shuffle(dataset.items)

    dataset_1 = dataset.sample(lambda items: items[:int(proportion * len(items))])
    dataset_2 = dataset.sample(lambda items: items[int(proportion * len(items)):])

    return dataset_1, dataset_2


def generate_candidate_tiles_from_items(items_dataset, sliding_windows, n_jobs=1):
    """
        High level helper function
        Apply a sliding window over each satellite image
         to generate a list of tiles (= regions of interest) to sample from
    Args:
        sliding_windows(list[SlidingWindow]|SlidingWindow):
        items_dataset(Dataset):
        n_jobs(int):

    Returns:
        dataset (Dataset):


    """
    LOGGER.info("Generating a pool of candidates tiles")
    tiles_dataset = Dataset(items=[])
    if not isinstance(sliding_windows, (list, tuple)):
        sliding_windows = [sliding_windows]
    for sliding_window in sliding_windows:
        LOGGER.info(sliding_window)
        tiles_dataset = tiles_dataset.extend(
            items_dataset.flatmap(sliding_window, desc="Applying sliding window", n_jobs=n_jobs))

    tiles_dataset = tiles_dataset.sample(lambda items: list(set(items)))
    LOGGER.info("State of dataset")
    LOGGER.info(roi_list_utils.get_state(tiles_dataset.items))
    return tiles_dataset


def sample_tiles_from_candidates(tiles_dataset, tiles_samplers):
    """
        High level helper function
        Apply a sampler over each satellite image's candidate tiles
         to generate a list of tiles (= regions of interest)
    Args:
        tiles_samplers(list[TilesSampler]|TilesSampler):
        tiles_dataset(Dataset)

    Returns:
        sampled_dataset(Dataset)
    """
    sampled_dataset = Dataset(items=[])
    LOGGER.info("Sampling tiles")
    if not isinstance(tiles_samplers, (list, tuple)):
        tiles_samplers = [tiles_samplers]

    for tiles_sampler in tiles_samplers:
        LOGGER.info(tiles_sampler)
        sampled_dataset = sampled_dataset.extend(tiles_dataset.sample(tiles_sampler))
        LOGGER.info("Tiles sampled, now generate the dataset using Dataset.generate_tiles_dataset")

    LOGGER.info(roi_list_utils.get_state(sampled_dataset.items))

    return sampled_dataset


def dump_dataset_tiles(tiles_dataset, items_dataset, output_dir, remove_first=False, save_format="jpg"):
    """
        High level helper function
        Actually generates training images from the dataset.sampled_tiles (= regions of interest)
        The filestructure is compatible with keras.ImageDataGenerator.flow_from_directory() method

        For more information on how to parse this, check this script:

        https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

        In summary, this is our directory structure:

        ```markdown
        output_dir/
            aircrafts/
                ac001.jpg
                ac002.jpg
                ...
            background/
                bg001.jpg
                bg002.jpg
                ...
        ```

    Args:
        items_dataset (Dataset):
        tiles_dataset (Dataset):
        output_dir(str): the output path
        save_format: "jpg" the image format
        remove_first(bool): erase output dir first?

    Returns:

    """
    LOGGER.info("Generating a dataset of tiles at location {}".format(output_dir))
    if remove_first:
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass

    def _dump_tiles(item):
        LOGGER.info("Dumping for item {}".format(item.key))
        tiles_dumper = ItemTileDumper(item, output_dir=output_dir, save_format=save_format)
        tiles_dataset_ = tiles_dataset.filter(lambda tile: tile.item_id == item.key, desc="Filtering")
        tiles_dataset_ = tiles_dataset_.map(tiles_dumper, desc="Saving tiles to {}".format(output_dir))

        return tiles_dataset_.items

    dumped_tiles = items_dataset.flatmap(_dump_tiles)

    return dumped_tiles
