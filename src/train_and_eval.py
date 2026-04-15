#!/usr/bin/env python3

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataloader import CLASSES, load_image_label_pairs, split_by_subject, ASLDataset
from .classifer import train_cnn, evaluate_cnn, evaluation_report


def main():
    print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7]
    )
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_image_label_pairs(args.data)
    train_pairs, val_pairs = split_by_subject(
        pairs, list(args.train_subjects), list(args.val_subjects)
    )

    results = {
        "config": {
            "data": args.data,
            "train_subjects": args.train_subjects,
            "val_subjects": args.val_subjects,
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "n_classes": len(CLASSES),
            "epochs": args.epochs,
        }
    }
    print(json.dumps(results))


if __name__ == "__main__":
    main()
