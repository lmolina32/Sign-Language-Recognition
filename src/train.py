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
from sklearn.metrics import confusion_matrix

from .dataloader import CLASSES, load_image_label_pairs, split_by_subject, ASLDataset
from .classifer import (
    train_cnn,
    evaluate_cnn,
    evaluation_report,
    extract_features,
    SVMClassifer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7]
    )
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-input", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--svm", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp",
        type=lambda s: s.lower() in {"1", "true", "yes"},
        default=True,
        help="Mixed precision on CUDA (no-op on CPU/MPS).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_image_label_pairs(args.data)
    train_pairs, val_pairs = split_by_subject(
        pairs, list(args.train_subjects), list(args.val_subjects)
    )

    # Train svm
    if args.svm:
        X_train, y_train = extract_features(
            train_pairs, target_size=(args.image_input, args.image_input)
        )
        print("Training SVC")
        svm = SVMClassifer(C=10.0, use_pca=True, n_components=300)
        svm.fit(X_train, y_train)
        svm.save(out / "svm_model.pkl")
    # Train CNN
    else:
        from .classifer import pick_device

        device = pick_device()
        print(f"device: {device}  amp={args.amp and device.type == 'cuda'}")

        train_ds = ASLDataset(
            train_pairs, augment=True, cnn_input_size=args.image_input
        )
        val_ds = ASLDataset(val_pairs, augment=False, cnn_input_size=args.image_input)
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=args.num_workers > 0,
        )

        print(f"Training CNN for {args.epochs} epochs")
        t0 = time.time()

        model, history = train_cnn(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=len(CLASSES),
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            verbose=args.verbose,
            out=out,
            use_amp=args.amp,
        )
        print(f"  total training time: {time.time()-t0:.1f}s")

        torch.save(model.state_dict(), out / "cnn_model.pt")
        print(f"saved cnn_model.pt")
        model_history = {"history": history}

        with open(out / "cnn_history.json", "w") as f:
            json.dump(model_history, f)
        print(f"\nModel doing training and cnn histories saved")


if __name__ == "__main__":
    main()
