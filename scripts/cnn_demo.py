#!/usr/bin/env python3
"""Run the full sign-language pipeline on a single image and print the prediction.

Example:
    python -m scripts.cnn_demo sample_data/P1_B_52.jpg --cnn-path results/final_cnn_model.pkl
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifer import CNN
from src.dataloader import CLASSES
from src.pipeline import Preprocessor


def run(image_path: Path, cnn_path: Path, image_size: int) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"loading model from {cnn_path}")
    # load model
    model = CNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(cnn_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # preprocess image
    preprocessor = Preprocessor()
    x = preprocessor.preprocess_final(img, target_size=(image_size, image_size))

    # Normalize
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    with torch.no_grad():
        x = x.to(device)
        x = x.unsqueeze(0)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()

    print(f"Predicted class index: {pred[0]}  ->  '{CLASSES[pred[0]]}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a sample image.")
    parser.add_argument(
        "--cnn-path",
        type=Path,
        default=Path("results/models/final_cnn_model.pth"),
        help="Saved CNN pipeline (.pth). Prediction is skipped if missing.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()
    run(args.image, args.cnn_path, args.image_size)


if __name__ == "__main__":
    main()
