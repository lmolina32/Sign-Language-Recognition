#!/usr/bin/env python3
"""Pick a random image from sample_data/test_data/ and run both classifiers on it.

This satisfies the Project 5 requirement that the program "pick one random sample
from the test set and present the processing result". The two per-image demos
(scripts/cnn_demo.py and scripts/svm_demo.py) remain unchanged and accept a
specific image path; this script is purely additive.

Example:
    python scripts/random_test_demo.py
    python scripts/random_test_demo.py --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifer import CNN, SVMClassifer
from src.dataloader import CLASSES
from src.pipeline import FeatureExtraction, Preprocessor, Segmentation


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DIR = REPO_ROOT / "sample_data" / "test_data"
DEFAULT_SVM_PATH = REPO_ROOT / "results" / "models" / "svm_model.pkl"
DEFAULT_CNN_PATH = REPO_ROOT / "results" / "models" / "final_cnn_model.pth"


def parse_ground_truth(image_path: Path) -> str:
    """Filenames are P{subject}_{class}_{n}.jpg — pull the class character."""
    try:
        return image_path.stem.split("_")[1].upper()
    except IndexError:
        return "?"


def run_svm(image: np.ndarray, svm_path: Path, image_size: int) -> str:
    if not svm_path.exists():
        return f"(svm model not found at {svm_path})"
    preprocessor = Preprocessor()
    segmentation = Segmentation()
    features = FeatureExtraction(pixels_per_cell=(16, 16))
    prep = preprocessor.preprocess_final(image, target_size=(image_size, image_size))
    seg = segmentation.segment(prep)
    feats = features.extract_all(
        prep, seg["contour"], seg["ycrcb_mask"], hand_bbox_crop=True
    )
    svm = SVMClassifer().load(str(svm_path))
    X = feats["features"].astype(np.float32).reshape(1, -1)
    pred_idx = int(svm.predict(X)[0])
    return CLASSES[pred_idx]


def run_cnn(image: np.ndarray, cnn_path: Path, image_size: int) -> str:
    if not cnn_path.exists():
        return f"(cnn model not found at {cnn_path})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(cnn_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = Preprocessor()
    x = preprocessor.preprocess_final(image, target_size=(image_size, image_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(1).cpu().numpy()[0])
    return CLASSES[pred_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick a random test image and run both classifiers on it."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--svm-path", type=Path, default=DEFAULT_SVM_PATH)
    parser.add_argument("--cnn-path", type=Path, default=DEFAULT_CNN_PATH)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    candidates = [
        p
        for p in args.test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not candidates:
        raise SystemExit(f"No test images found in {args.test_dir}")

    pick = random.choice(candidates)
    truth = parse_ground_truth(pick)
    image = cv2.imread(str(pick))
    if image is None:
        raise SystemExit(f"Could not read image: {pick}")

    print(f"Picked random test image: {pick.name}")
    print(f"  ground truth: '{truth}'")
    print(f"  image shape : {image.shape}")

    svm_pred = run_svm(image, args.svm_path, args.image_size)
    cnn_pred = run_cnn(image, args.cnn_path, args.image_size)

    print(f"  SVM prediction : '{svm_pred}'   {'OK' if svm_pred == truth else 'X'}")
    print(f"  CNN prediction : '{cnn_pred}'   {'OK' if cnn_pred == truth else 'X'}")


if __name__ == "__main__":
    main()
