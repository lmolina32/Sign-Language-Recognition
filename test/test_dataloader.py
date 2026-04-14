#!/usr/bin/env python3

import pytest

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_image_label_pairs, CLASSES_TO_IDX


def test_load_image_label_pairs() -> None:
    cwd = Path.cwd().parts
    if cwd[-1] == "test":
        sample_data = Path.cwd() / "../sample_data"
    else:
        sample_data = Path.cwd() / "/sample_data"

    pairs = load_image_label_pairs(str(sample_data))
    assert pairs is not None
    img_path = sample_data / "P1_B_52.jpg"
    assert (str(img_path), CLASSES_TO_IDX["B"]) in pairs
