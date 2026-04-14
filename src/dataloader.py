#!/usr/bin/env python3

from pathlib import Path
from typing import Tuple, List

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASSES_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_image_label_pairs(root_dir: str) -> List[Tuple[str, int]]:
    """Load (image_path, label_idx) pairs from a directory

    Args:
        root_dir (str): directory of data

    Expected layout:
        roo_dir/
            P1_A_001.jpg
            ...
    """
    pairs = []
    valid_extensions = {".jpg", ".jpeg", ".png"}
    for fp in Path(root_dir).iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in valid_extensions:
            raise ValueError("Expected files with valid extension (.jpg, .jpeg, .png)")
        try:
            img_class = fp.stem.split("_")[1].upper()
        except (KeyError, IndexError, AttributeError):
            raise ValueError(
                "Expected files of the format p{paritcipant number}_{img_class}_{img_number}"
            )
        if img_class not in CLASSES_TO_IDX:
            raise ValueError("Expected class must be from 0-9 or A-Z")
        pairs.append((str(fp), CLASSES_TO_IDX[img_class]))
    return pairs
