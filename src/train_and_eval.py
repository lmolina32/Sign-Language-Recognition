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
from .classifer import train_cnn, evaluate_cnn, evaluation_report


def main():
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
            "image_dim": args.image_input,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_subjects": args.train_subjects,
            "val_subjects": args.val_subjects,
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "n_classes": len(CLASSES),
            "epochs": args.epochs,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = ASLDataset(train_pairs, augment=True, cnn_input_size=args.image_input)
    val_ds = ASLDataset(val_pairs, augment=False, cnn_input_size=args.image_input)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
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
    )
    print(f"  total training time: {time.time()-t0:.1f}s")

    train_eval_ds = ASLDataset(
        train_pairs, augment=False, cnn_input_size=args.image_input
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    y_train_pred, y_train_true, _ = evaluate_cnn(model, train_eval_loader, device)
    y_val_pred, y_val_true, _ = evaluate_cnn(model, val_loader, device)

    train_metrics = evaluation_report(y_train_true, y_train_pred)
    val_metrics = evaluation_report(y_val_true, y_val_pred)

    print(
        f"\nCNN train accuracy: {train_metrics['accuracy']:.4f}  "
        f"({train_metrics['n_correct']}/{train_metrics['n_total']})"
    )
    print(
        f"CNN val   accuracy: {val_metrics['accuracy']:.4f}  "
        f"({val_metrics['n_correct']}/{val_metrics['n_total']})"
    )
    print(f"CNN val   macro F1: {val_metrics['macro_f1']:.4f}")

    results["cnn"] = {"train": train_metrics, "val": val_metrics, "history": history}

    cm = confusion_matrix(y_train_true, y_val_pred, labels=list(range(len(CLASSES))))
    np.save(out / "cnn_confusion_matrix.npy", cm)
    torch.save(model.state_dict(), out / "cnn_model.pt")
    print(f"saved cnn_model.pt, cnn_confusion_matrix.npy")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
