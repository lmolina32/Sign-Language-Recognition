#!/usr/bin/env python3

import pytest

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_image_label_pairs, split_by_subject, CLASSES_TO_IDX


@pytest.fixture
def data_path() -> Path:
    cwd = Path.cwd().parts
    if cwd[-1] == "test":
        sample_data = Path.cwd() / "../sample_data"
    else:
        sample_data = Path.cwd() / "/sample_data"
    return sample_data


@pytest.fixture
def download_data_path() -> Path:
    cwd = Path.cwd().parts
    if cwd[-1] == "test":
        sample_data = Path.cwd() / "../data"
    else:
        sample_data = Path.cwd() / "/data"
    return sample_data


def test_load_image_label_pairs(data_path) -> None:
    pairs = load_image_label_pairs(str(data_path))
    assert pairs is not None
    img_path = data_path / "P1_B_52.jpg"
    assert (str(img_path), CLASSES_TO_IDX["B"]) in pairs


def test_load_image_label_pairs_on_download(download_data_path) -> None:
    if download_data_path.exists():
        train_path = download_data_path / "train"
        test_path = download_data_path / "test"
        if not train_path.exists() or not test_path.exists():
            return
        pairs = load_image_label_pairs(str(download_data_path))
        assert pairs is not None
        assert len(pairs) == 36000


def test_split_by_subject(data_path) -> None:
    pairs = load_image_label_pairs(str(data_path))
    trains, vals = split_by_subject(pairs, [1], [3])

    for path, _ in trains:
        image_path = Path(path).parts[-1]
        image_suffix = Path(image_path).stem
        id = image_suffix.split("_")[0][1:]
        assert int(id) == 1

    for path, _ in vals:
        image_path = Path(path).parts[-1]
        image_suffix = Path(image_path).stem
        id = image_suffix.split("_")[0][1:]
        assert int(id) == 3
