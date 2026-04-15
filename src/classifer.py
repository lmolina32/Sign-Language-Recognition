#!/usr/bin/env python3

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def train_cnn(
    train_loader,
    val_loader,
    num_classes=36,
    epochs=15,
    lr=1e-3,
    device=None,
    verbose=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
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
