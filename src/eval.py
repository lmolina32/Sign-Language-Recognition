#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .dataloader import CLASSES, load_image_label_pairs, split_by_subject, ASLDataset
from .classifer import (
    evaluate_cnn,
    evaluation_report,
    extract_features,
    SVMClassifer,
    CNN,
)


def evaluate_svm(args, train_pairs, val_pairs, out):
    results = {"kernel": "rbf", "pca": "true", "C": 10.0}
    print("loading the val and trainings sets...")
    X_train, y_train = extract_features(
        train_pairs, target_size=(args.image_input, args.image_input)
    )
    X_val, y_val = extract_features(
        val_pairs, target_size=(args.image_input, args.image_input)
    )

    print("loading svm...")
    svm = SVMClassifer(C=10.0, use_pca=True, n_components=300)
    svm.load(args.svm_path)

    # evaluate SVM
    y_train_pred = svm.predict(X_train)
    train_metrics = evaluation_report(y_train, y_train_pred)
    y_val_pred = svm.predict(X_val)
    val_metrics = evaluation_report(y_val, y_val_pred)

    print(
        f"\nSVM train accuracy: {train_metrics['accuracy']:.4f}  "
        f"({train_metrics['n_correct']}/{train_metrics['n_total']})"
    )
    print(
        f"SVM val   accuracy: {val_metrics['accuracy']:.4f}  "
        f"({val_metrics['n_correct']}/{val_metrics['n_total']})"
    )
    print(f"SVM val   macro F1: {val_metrics['macro_f1']:.4f}")
    results["svm"] = {"train": train_metrics, "val": val_metrics}
    cm = confusion_matrix(y_val, y_val_pred, labels=list(range(len(CLASSES))))
    np.save(out / "svm_confusion_matrix.npy", cm)

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )

    plt.title("SVM Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(out / "svm_confusion_matrix_plot.png", dpi=300, bbox_inches="tight")

    del X_train, X_val, y_train, y_val, svm
    gc.collect()

    # save metrics
    with open(out / "svm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved to {out / 'svm_metrics.json'}")


def evalute_cnn_saved_model_train_val(
    args,
    train_pairs,
    val_pairs,
    results,
    out,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading model from {args.cnn_path}")
    model = CNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(args.cnn_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("loading train and val datasets")
    train_eval_ds = ASLDataset(
        train_pairs, augment=False, cnn_input_size=args.image_input
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    val_ds = ASLDataset(val_pairs, augment=False, cnn_input_size=args.image_input)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("Running training evalution of cnn")
    y_train_pred, y_train_true, _ = evaluate_cnn(model, train_eval_loader, device)
    print("Running validation evalution of cnn")
    y_val_pred, y_val_true, _ = evaluate_cnn(model, val_loader, device)

    print("creating reports")
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

    results["cnn"] = {
        "train": train_metrics,
        "val": val_metrics,
    }

    with open(out / "train_val_cnn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved to {out / 'train_val_cnn_metrics.json'}")

    cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(len(CLASSES))))
    np.save(out / "val_cnn_confusion_matrix.npy", cm)
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )

    plt.title("CNN Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(out / "cnn_confusion_matrix_plot.png", dpi=300, bbox_inches="tight")


def evaluate_cnn_saved_model_on_test(args, pairs, results, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading model from {args.cnn_path}")
    model = CNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(args.cnn_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("loading test dataset")
    test_ds = ASLDataset(pairs, augment=False, cnn_input_size=args.image_input)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("Running test evalution of cnn")
    y_test_pred, y_test_true, _ = evaluate_cnn(model, test_loader, device)

    print("creating reports")
    test_metrics = evaluation_report(y_test_true, y_test_pred)

    print(
        f"CNN test   accuracy: {test_metrics['accuracy']:.4f}  "
        f"({test_metrics['n_correct']}/{test_metrics['n_total']})"
    )
    print(f"CNN val   macro F1: {test_metrics['macro_f1']:.4f}")

    results["cnn"] = {"test": test_metrics}

    with open(out / "test_cnn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved to {out / 'test_cnn_metrics.json'}")

    cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(len(CLASSES))))
    np.save(out / "test_cnn_confusion_matrix.npy", cm)
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )

    plt.title("CNN Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(
        out / "test_cnn_confusion_matrix_plot.png", dpi=300, bbox_inches="tight"
    )


def parse_arguments() -> argparse.Namespace:
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
    parser.add_argument(
        "--svm-path", type=str, help="SVM path (required if svm is set to true)"
    )
    parser.add_argument(
        "--cnn-path",
        type=str,
        help="CNN model path (required if svm is set to false (default ))",
    )
    parser.add_argument(
        "--cnn-test", type=bool, default=False, help="running test data on cnn"
    )

    args = parser.parse_args()

    if args.svm and not args.svm_path:
        parser.error("--svm requires --svm-path to load model")

    if not args.svm and not args.cnn_path:
        parser.error("--cnn-path required if --svm was not set to True")

    return args


def main():
    args = parse_arguments()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_image_label_pairs(args.data)
    if not args.cnn_test:
        train_pairs, val_pairs = split_by_subject(
            pairs, list(args.train_subjects), list(args.val_subjects)
        )

    results = {
        "config": {
            "data": args.data,
            "image_dim": args.image_input,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_classes": len(CLASSES),
            "epochs": args.epochs,
        }
    }

    if args.svm:
        evaluate_svm(args, train_pairs, val_pairs, out)
    elif not args.svm and not args.cnn_test:
        results["train_subjects"] = args.train_subjects
        results["val_subjects"] = args.val_subjects
        results["n_train"] = len(train_pairs)
        results["n_val"] = len(val_pairs)
        evalute_cnn_saved_model_train_val(args, train_pairs, val_pairs, results, out)
    elif not args.svm and args.cnn_test:
        results["subjects"] = 2
        evaluate_cnn_saved_model_on_test(args, pairs, results, out)


if __name__ == "__main__":
    main()
