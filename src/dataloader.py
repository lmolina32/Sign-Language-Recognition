#!/usr/bin/env python3

from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .pipeline import Preprocessor

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASSES_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

Pairs = List[Tuple[str, int]]


def load_image_label_pairs(root_dir: str) -> Pairs:
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


def split_by_subject(pairs, train_subjects, val_subjects) -> Tuple[Pairs, Pairs]:
    """Split training and validation based on subjects

    Args:
        train_subjects (list[int]): list of training subjects by number
        val_subjects (list[int]): list of validators subjects by number
    """
    train, val = [], []

    for path, label in pairs:
        image_path = Path(path).parts[-1]
        image_suffix = Path(image_path).stem
        try:
            participant_id = int(image_suffix.split("_")[0][1:])
        except (KeyError, IndentationError, IndexError):
            raise ValueError(
                "Expected files of the format p{paritcipant number}_{img_class}_{img_number}"
            )
        if participant_id in train_subjects:
            train.append((str(path), label))
        elif participant_id in val_subjects:
            val.append((str(path), label))
    return train, val


class ASLDataset(Dataset):
    def __init__(
        self,
        pairs: Pairs,
        preprocessor: Optional[Preprocessor] = None,
        augment: bool = False,
        cnn_input_size: int = 96,
    ) -> None:
        self.pairs: Pairs = pairs
        self.preprocessor: Preprocessor = preprocessor or Preprocessor()
        self.augment: bool = augment
        self.cnn_input_size = cnn_input_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, key):
        path, label = self.pairs[key]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        prep = self.preprocessor.preprocess(img)
        x = prep["final"]

        # resize image for smaller CNNs
        if self.cnn_input_size != 224:
            x = cv2.resize(
                x,
                (self.cnn_input_size, self.cnn_input_size),
                interpolation=cv2.INTER_AREA,
            )

        # augment the data if flag passed in
        if self.augment:
            if np.random.random() < 0.5:
                x = cv2.flip(x, 1)
            if np.random.random() < 0.5:
                jitter = np.random.uniform(0.85, 1.15)
                x = np.clip(x.astype(np.float32) * jitter, 0, 255).astype(np.uint8)

        # BGR -> RGB -> normalize -> center around 0 -> convert to torch tensor
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return x, label
