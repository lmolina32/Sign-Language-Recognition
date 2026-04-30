#!/usr/bin/env python3
"""Real time sign language recongnition from a webcame feed"""

import sys
import time
from pathlib import Path

import argparse
import cv2
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifer import CNN
from src.dataloader import CLASSES
from src.pipeline import Preprocessor


def load_model(cnn_path: Path, device: torch.device) -> CNN:
    """Load CNN from .pth checkpoint and set to eval mode"""
    print(f"loading model from {cnn_path} to {device}")
    model = CNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(cnn_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded and ready")
    return model


def predict(
    frame: np.ndarray,
    model: CNN,
    preprocessor: Preprocessor,
    device: torch.device,
    image_size: int,
) -> tuple[str, float]:
    """Return (class_label, confidence) for a single BGR frame."""
    x = preprocessor.preprocess_final(frame, target_size=(image_size, image_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)

    label = CLASSES[idx.item()]
    confidence = conf.item()
    return label, confidence


def draw_overlay(
    frame: np.ndarray, label: str, confidence: float, last_inference_ms: float
) -> np.ndarray:
    """Burn prediction text onto the frame and return it."""
    h, w = frame.shape[:2]

    # Semi-transparent banner at the top
    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = f"{label}  ({confidence * 100:.1f}%)"
    cv2.putText(
        frame,
        text,
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )

    # Small timing note bottom-right
    timing = f"{last_inference_ms:.0f} ms"
    cv2.putText(
        frame,
        timing,
        (w - 100, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


def run(
    cnn_path: Path, image_size: int, camera_index: int, inference_interval: float
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cnn_path, device)
    preprocessor = Preprocessor()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")

    print("Press  q  or  Esc  to quit.")

    label = "-"
    confidence = 0.0
    last_inference_ms = 0.0
    next_inference_at = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, exiting.")
            break

        now = time.monotonic()
        if now >= next_inference_at:
            t0 = time.monotonic()
            try:
                label, confidence = predict(
                    frame, model, preprocessor, device, image_size
                )
            except Exception as exc:
                print(f"Inference error: {exc}")
                label = "error"
                confidence = 0.0
            last_inference_ms = (time.monotonic() - t0) * 1000
            next_inference_at = now + inference_interval

        display = draw_overlay(frame.copy(), label, confidence, last_inference_ms)
        cv2.imshow("Sign Language Recognition  -  q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time sign-language CNN inference from webcam."
    )
    parser.add_argument(
        "--cnn-path",
        type=Path,
        default=Path("results/models/final_cnn_model.pth"),
        help="Path to the saved CNN checkpoint (.pth).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Square size (px) the model expects (default: 96).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between inference calls (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.cnn_path, args.image_size, args.camera, args.interval)


if __name__ == "__main__":
    main()
