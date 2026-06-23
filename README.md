# Sign Language Recognition

**Team:** Leonardo Molina, Alphonsus Koong Bok Hui
**Course:** CSE 40535 Computer Vision, Spring 2026

Classifying static hand-sign images with both a classical SVM and a CNN.

## Repository Structure

```text
Sign-Language-Recognition/
├── docs/                # Project assignments (PDFs) and per-project write-ups
├── results/             # Trained models and evaluation outputs
├── sample_data/         # Sample images, including the held-out test set
├── scripts/             # Dataset download and demos
├── src/                 # Project source code
│   ├── classifer.py     # SVM and CNN model classes
│   ├── dataloader.py    # Data loading helpers
│   ├── eval.py          # Evaluation for SVM and CNN
│   ├── pipeline.py      # Image preprocessing pipeline
│   └── train.py         # Training for SVM and CNN
├── test/                # Unit tests
└── requirements.txt     # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_dataset.sh   # download the dataset from source
```

## Demos

Run a single test image through either classifier:

```bash
python scripts/cnn_demo.py sample_data/test_data/P11_2_143.jpg --cnn-path results/cnn_results/final_cnn_model.pkl
python scripts/svm_demo.py sample_data/test_data/P11_2_143.jpg --svm-path results/svm_results/svm_model.pkl
```

Real-time recognition from a webcam feed:

```bash
python scripts/cnn_camera.py
```

Pick one random sample from the held-out test set and run both classifiers on it:

```bash
python scripts/random_test_demo.py
# or, for a reproducible pick:
python scripts/random_test_demo.py --seed 42
```

## Project Updates

- [Project 3 update](docs/project03_update.md)
- [Project 4 update](docs/project04_update.md)
- [Project 5 (final) update](docs/project05_update.md) — final report: test-set description, accuracy, analysis of the val→test drop, NN vs. classical comparison, and proposed improvements.

## Presentation

Project presentation: [Google Drive](https://drive.google.com/file/d/1DI7TsgrMnAGYE0NXD8tr-18etQMK0ZW-/view?usp=sharing).
