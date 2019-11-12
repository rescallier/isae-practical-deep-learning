"""
Downloading data
"""
import os
import subprocess

from khumeia import LOGGER

ROOT_URL = "https://storage.googleapis.com/isae-deep-learning"


def _download_data(archive="tp_isae_data.tar.gz", data_dir=None, check_dir=None):
    """

    Args:
        archive:
        data_dir:

    Returns:

    """
    assert data_dir is not None, "please specify a download dir or better specify TP_DATA env variable"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    LOGGER.info("Downloading data from {} to {}".format(archive, data_dir))

    if not os.path.exists(os.path.join(data_dir, archive)):
        # Download tar gz
        LOGGER.info("Downloading {}".format("{}/{}").format(ROOT_URL, archive))
        cmd = ["curl", "-X", "GET", "{}/{}".format(ROOT_URL, archive), "--output", os.path.join(data_dir, archive)]
        subprocess.check_call(cmd)
    if check_dir is None or not os.path.exists(os.path.join(data_dir, check_dir)):
        # Untar it
        LOGGER.info("Extracting tar gz")
        cmd = ["tar", "-zxvf", os.path.join(data_dir, archive), "-C", data_dir]
        subprocess.check_call(cmd)


def download_train_data(data_dir=None):
    """
    Download the raw training data to data dir and extracts
    Args:
        data_dir:

    Returns:

    """
    data_dir = data_dir or os.path.expandvars(os.environ.get("TP_DATA"))
    LOGGER.info("Downloading training data")
    _download_data(archive="tp_isae_train_data.tar.gz", data_dir=data_dir, check_dir="raw/trainval")
    LOGGER.info("Done. Your training data is located here {}\n".format(os.path.join(data_dir, "raw", "trainval")))


def download_eval_data(data_dir=None):
    """
    Download the raw eval data to data dir and extracts
    Args:
        data_dir:

    Returns:

    """
    data_dir = data_dir or os.path.expandvars(os.environ.get("TP_DATA"))
    LOGGER.info("Downloading evaluation data")
    _download_data(archive="tp_isae_eval_data.tar.gz", data_dir=data_dir, check_dir="raw/eval")
    LOGGER.info("Done. Your data is located here {}\n".format(os.path.join(data_dir, "raw", "eval")))
