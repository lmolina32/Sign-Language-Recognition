#!/usr/bin/env python3

import gc
import numpy as np
import pickle
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from .pipeline import Preprocessor, Segmentation, FeatureExtraction


def extract_features(
    pairs,
    pixels_per_cell=(16, 16),
    target_size=(224, 224),
    hand_bbox_crop: bool = True,
):
    """Build an (N, D) feature matrix from (path, label) pairs.

    `hand_bbox_crop=True` (the Part 4 small-improvement setting) crops to the
    segmentation bounding box before HOG so background pixels don't leak into the
    descriptor. Set to False to reproduce the original full-image HOG path.
    """
    preprocessor = Preprocessor()
    segmentation = Segmentation()
    featureextraction = FeatureExtraction(pixels_per_cell=pixels_per_cell)

    X, y = [], []
    for i, (path, label) in enumerate(pairs):
        img = cv2.imread(path)
        if img is None:
            continue
        prep_img = preprocessor.preprocess_final(img, target_size=target_size)
        seg = segmentation.segment(prep_img)
        feats = featureextraction.extract_all(
            prep_img,
            seg["contour"],
            seg["ycrcb_mask"],
            hand_bbox_crop=hand_bbox_crop,
        )
        X.append(feats["features"].astype(np.float32))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class SVMClassifer:
    def __init__(self, C=10.0, use_pca=True, n_components=300) -> None:
        steps = [("scaler", StandardScaler(with_mean=False))]
        if use_pca:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        random_state=42,
                        svd_solver="randomized",
                    ),
                )
            )
        steps.append(
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma="scale",
                    random_state=42,
                    verbose=True,
                    max_iter=1000,
                ),
            )
        )

        self.model = Pipeline(steps)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


class CNN(nn.Module):

    def __init__(self, num_classes: int = 36) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 28 -> 14
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def pick_device():
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_cnn(
    train_loader,
    val_loader,
    out,
    num_classes=36,
    epochs=15,
    lr=1e-3,
    device=None,
    verbose=True,
    use_amp: bool = True,
):
    if device is None:
        device = pick_device()

    model = CNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        print(f"\n[Epoch {epoch+1}/{epochs}] Starting training...")
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            if batch_idx % 10 == 0:
                print(
                    f"  [Epoch {epoch+1}, Batch {batch_idx}] "
                    f"loss={total_loss/total:.4f}, acc={correct/total:.4f}, "
                    f"samples_seen={total}"
                )
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        print(f"  [Epoch {epoch+1}] Running validation...")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
                if batch_idx % 10 == 0:
                    print(
                        f"  [Val Batch {batch_idx}] "
                        f"running_loss={v_loss/v_total:.4f}, acc={v_correct/v_total:.4f}"
                    )
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            out / f"checkpoint_epoch_{epoch+1}.pth",
        )

        print(f"  Checkpoint saved for epoch {epoch+1}")
        scheduler.step()

        if verbose:
            print(
                f"  Epoch {epoch+1:2d}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        gc.collect()
    return model, history


def evaluate_cnn(model, loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels, accuracy_score(labels, preds)


def evaluation_report(y_true, y_pred):
    """Compute accuracy + macro/weighted P/R/F."""
    acc = accuracy_score(y_true, y_pred)
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "n_correct": int((y_true == y_pred).sum()),
        "n_total": int(len(y_true)),
    }
