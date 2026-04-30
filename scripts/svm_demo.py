#!/usr/bin/env python3
"""Run the full sign-language pipeline on a single image and print the prediction.

Example:
    python -m scripts.svm_demo sample_data/P1_B_52.jpg --svm-path results/svm_model.pkl
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifer import SVMClassifer
from src.dataloader import CLASSES
from src.pipeline import FeatureExtraction, Preprocessor, Segmentation


def run(image_path: Path, svm_path: Path | None, image_size: int) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    preprocessor = Preprocessor()
    segmentation = Segmentation()
    features = FeatureExtraction(pixels_per_cell=(16, 16))

    prep = preprocessor.preprocess_final(img, target_size=(image_size, image_size))
    seg = segmentation.segment(prep)
    feats = features.extract_all(
        prep, seg["contour"], seg["ycrcb_mask"], hand_bbox_crop=True
    )

    print(f"Image: {image_path}")
    print(f"HOG feature length: {feats['hog_feats'].shape[0]}")
    print(f"Contour descriptors: {feats['contour_dict']}")

    if svm_path is None or not Path(svm_path).exists():
        print("(no SVM model path given or file missing — printing features only)")
        return

    svm = SVMClassifer().load(str(svm_path))
    X = feats["features"].astype(np.float32).reshape(1, -1)
    pred = int(svm.predict(X)[0])
    print(f"Predicted class index: {pred}  ->  '{CLASSES[pred]}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a sample image.")
    parser.add_argument(
        "--svm-path",
        type=Path,
        default=Path("results/models/svm_model.pkl"),
        help="Saved SVM pipeline (.pkl). Prediction is skipped if missing.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()
    run(args.image, args.svm_path, args.image_size)


if __name__ == "__main__":
    main()
