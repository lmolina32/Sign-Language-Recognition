# Part 5: Final Update — Testing on Unknown Data

This is the final milestone of the project. We tested the two classifiers we built in
Part 4 — an RBF-kernel SVM on hand-crafted features and a small custom CNN — against a
held-out test set the models had never seen during design or training.

## Test database

**Source.** We collected the test set ourselves on our own smartphones — it is *not*
drawn from the Mendeley dataset that supplied training and validation data. Each team
member shot **5 images per class across all 36 classes**, giving a target of 180 images
per subject and **360 images total** (`5 × 36 × 2`). The two test subjects are
labelled to keep them disjoint from the Mendeley participant IDs (P1–P10, used for
train + val):

- **P11 = Al** (180 images, 5 per class × 36 classes)
- **P20 = Leo** (180 images, 5 per class × 36 classes)

Of the 360 we shot, **10 were pulled out as a visual sample** and checked into the
repository at `sample_data/test_data/` (so reviewers can see the test distribution
without downloading the full set), leaving **350 images for the evaluation run** that
produced the metrics in `results/{svm,cnn}_results/test_*_metrics.json`
(`360 − 10 = 350`).

**What's different vs. train and validation.** This is the core reason the test set
exercises the program in a way train/val cannot. The Mendeley training data is a
*curated* corpus — pre-cropped, small JPEGs, consistent framing — and using its own
`test/` partition would only test against more of the same distribution. Our
self-collected smartphone test set introduces a real domain shift along three axes,
listed in order of how much they hurt our classifiers:

1. **Native resolution.** Mendeley training images are pre-cropped JPEGs of roughly
   **150×150 pixels** (~16 KB on disk). Our smartphone test images are raw photos at
   **3648×2736 or 4032×3024 pixels** (~1.5 MB on disk) — about 25× larger on each axis,
   ~600× more pixels. Our pipeline resizes everything to 96×96 with `INTER_AREA` before
   HOG and the CNN ever see it, but downsampling that aggressively is not the same as
   downsampling a tightly cropped training image: a finger that is 30 px wide in a
   150² training image becomes a ~2-px stripe in the 96² resize of a 4032² test
   image. HOG cells (16×16 px) and the first conv layer (3×3 kernels) can no longer
   resolve it.
2. **Framing and context.** Mendeley training images are tightly cropped — the hand
   fills 70–90% of the frame and the wrist is barely visible. Our test images are
   wide-frame phone photos where the hand fills only **30–50% of the frame**, with
   substantial visible wall background, **forearm and wrist exposed**, sometimes
   shoulder/sleeve at the frame edge, and a hard **shadow cast onto the wall** behind
   the hand. The YCrCb skin segmenter we tuned on training data picks up the forearm
   and (in some shots) the sleeve cuff, so the "hand contour" it returns is no longer
   hand-shaped.
3. **Subject novelty.** P11 and P20 are completely unseen, just like P8–P10 were
   unseen during training. Part 4 val numbers already capture the cost of subject
   generalization, so subject novelty alone is not what drops test accuracy from 75%
   to 12% — but combined with (1) and (2) it compounds.

**Why these differences are sufficient to test the final program.** The spec asks for
"unknown data" and the deployment scenario we care about is exactly this: a user picks
up their phone, holds out a hand, and snaps a picture. The Mendeley training data is a
curated, pre-cropped dataset; our test data is raw camera output collected by us with
no curation. If our pipeline only works on the curated distribution, that's not a
working classifier — it's a memorizer of the original photographer's cropping
conventions. The resolution + framing shift, combined with the subject novelty,
forces the program to confront that.

## Test accuracy

We re-evaluated the same trained checkpoints used in Part 4 against the unseen test set
using the same metrics (accuracy, macro/weighted precision/recall/F1). Raw JSON is at
`results/svm_results/test_svm_metrics.json` and `results/cnn_results/test_cnn_metrics.json`;
confusion matrices are at `results/{svm,cnn}_results/test_*_confusion_matrix_plot.png`.

| Classifier | Train acc. | Val acc. | Val macro-F1 | **Test acc.** | **Test macro-F1** |
| ---------- | ---------- | -------- | ------------ | ------------- | ----------------- |
| SVM (RBF, PCA-300, hand-bbox HOG) | 1.0000 | 0.7370 | 0.7308 | **0.1286** | **0.1237** |
| Small CNN (10 epochs)             | 0.9978 | 0.7773 | 0.7668 | **0.1229** | **0.0888** |

Random-chance on 36 classes is 1/36 ≈ 2.78%, so both classifiers are still ~4–5× above
chance — they have not collapsed entirely — but val→test is a **~60-point drop** in
absolute accuracy for both. The SVM and CNN end up tied to within one point of each
other on the test set despite the CNN's 4-point val advantage, which is the single most
informative finding of this milestone. We discuss why below.

## Why the test results are worse, with illustrations

The test-set drop is dominated by **image-domain shift**, not by the
unseen-subject effect alone. Two side-by-side examples make the point cleanly:

| | Training image (P3, class `S`) | Test image (P11, class `S`) |
| --- | --- | --- |
| Native resolution | ~150×150 px | 3648×2736 px |
| Hand fraction of frame | ~80% | ~35% |
| Forearm / wrist visible | minimal | wrist + sleeve cuff |
| Background | textured/patterned | smooth wall + hard shadow |
| Shape after 96×96 resize | ~75 px hand | ~33 px hand |

Both files are checked in: `sample_data/training_data/P3_S_234.jpg` (training)
and `sample_data/test_data/P11_S_094.JPG` (test). Visually, the hand silhouette is
broadly similar — both are closed fists viewed from the palm side — but the *gradient
content* HOG is asked to summarize is very different. In the training crop,
HOG's 16×16 cells each see ~3 fingers worth of detail; in the test resize, the same
cell sees 2 fingers plus a chunk of background wall plus a shadow edge. The SVM
trained on the first distribution doesn't have a feature in its vocabulary for the
second.

The CNN suffers in a related but distinct way. Its first conv layer learned 3×3 filters
on training-distribution gradients (sharp finger edges against textured backgrounds,
fingertip-sized features ~5–10 px). Test images, after the same resize, present
fingertip-sized features at ~2 px and large smooth wall regions the CNN never had to
ignore at training time. Confusion matrices confirm this: SVM test errors are spread
roughly uniformly with hot spots on visually-similar pairs (`0`/`O`, `M`/`N`, `2`/`V`),
while CNN test errors **collapse onto a few classes** (`L`, `T`, and `5` show up
disproportionately) — a sign that the CNN, robbed of the texture cues it used to
discriminate, falls back to a small set of "default" predictions.

Running `python scripts/random_test_demo.py` with three different seeds illustrates:

```
P20_A_001.JPG   truth='A'   SVM='M'  X     CNN='T'  X
P11_6_165.JPG   truth='6'   SVM='0'  X     CNN='5'  X
P11_Z_128.JPG   truth='Z'   SVM='0'  X     CNN='L'  X
```

In all three the SVM and CNN disagree, which is consistent with the two models having
collapsed onto *different* spurious cues rather than both learning the same wrong
thing.

### Improvements we'd make to lower the error rate

Listed cheapest first.

1. **Train-time augmentation that mimics the test distribution.** Random crops at
   25–80% of frame area, random rotations ±15°, random brightness/contrast/Gaussian-blur
   to simulate phone-camera variation, and random occlusion of the bottom edge to
   simulate forearm exposure. Adds nothing at inference time and would directly close
   the framing gap. We expect this single change to add 20–30 absolute points on test
   accuracy.
2. **Hand detector front-end.** Run a lightweight hand detector (MediaPipe Hands or a
   YOLOv8-nano fine-tuned for hands) on every input, crop to the predicted bbox, then
   feed the crop into our existing pipeline. This effectively normalizes both training
   and test inputs to the same "tightly cropped hand" distribution and is the single
   biggest fix available — it removes the resolution/framing axis entirely. Cost is
   one extra forward pass per image; on phone hardware that's still real-time.
3. **Multi-resolution training.** Currently every image goes through the same
   `Preprocessor.preprocess_final(target_size=(96, 96))`. Training on a mix of 64², 96²,
   and 128² and randomizing per batch teaches the CNN scale invariance directly.
4. **Pretrained backbone.** Replace the from-scratch 4-block CNN with an ImageNet-
   pretrained ResNet-18, freezing the first two blocks and fine-tuning the rest. The
   ImageNet prior is what carries pretrained models across domain shifts; our
   from-scratch CNN has no such prior. Per-epoch cost roughly doubles but should add
   significant test accuracy.
5. **Test-time augmentation.** At inference, average the prediction over horizontal
   flip and ±10° rotation. Almost free, usually 1–2 points.
6. **Skin-segmenter robustness.** The current YCrCb thresholds are fixed
   (`Cr ∈ [133, 173], Cb ∈ [77, 127]`, from Kovac et al. 2003). Lighting in the test
   set fires false positives on shadow regions and false negatives on the lighter
   knuckle highlights. Replacing the fixed thresholds with a small learned skin
   classifier (logistic regression on YCrCb pixels with a few thousand labelled
   pixels) would be more robust and is a 50-line change.

## Note 2 — neural network vs. classical comparison

This is the most interesting finding of Part 5. On val, the CNN beats the SVM by a
respectable 4 absolute points (77.7% vs 73.7%). On test, the gap **disappears and
slightly inverts** — SVM 12.86% vs CNN 12.29%. The CNN's val-set advantage came largely
from learning subject-correlated cues (skin tone, sleeve color, lighting profile) that
let it discriminate among held-out subjects who shared photographic conventions with
the training set; once those photographic conventions changed at test time, the
advantage evaporated. The SVM's hand-crafted features (HOG + Hu Moments + contour
descriptors) are pre-baked invariants — translation-, scale-, and rotation-tolerant
by construction — so they don't *adapt* to new distributions, but they also don't
*memorize* the old one. The result is that the more "hand-engineered for invariance"
classifier degrades more gracefully under domain shift, which lines up with what the
literature on classical-vs-deep features generally predicts at this data scale.

The lesson for our project specifically: any improvement that better aligns train and
test distributions (augmentation, hand-detector front-end, multi-resolution training)
should help the CNN much more than the SVM. The CNN has the capacity to use those
extra signals; the SVM, on the same features, will plateau at whatever HOG can
represent.

## How to run

One-time setup (~1.1 GB download):

```bash
pip install -r requirements.txt
bash scripts/download_dataset.sh
```

Train + evaluate the SVM:

```bash
python -m src.train --data data/all/ --svm True
python -m src.eval  --data data/all/ --svm True --svm-path results/models/svm_model.pkl
python -m src.eval  --data data/test/ --svm True --svm-test True \
    --svm-path results/models/svm_model.pkl
```

Train + evaluate the CNN (`--amp` is mixed precision on CUDA, no-op on MPS/CPU):

```bash
python -m src.train --data data/all/ --epochs 10 --batch-size 64 --num-workers 2 --amp True
python -m src.eval  --data data/all/  --cnn-path results/models/final_cnn_model.pth
python -m src.eval  --data data/test/ --cnn-test True \
    --cnn-path results/models/final_cnn_model.pth
```

Run a single saved test image through either classifier:

```bash
python scripts/svm_demo.py sample_data/test_data/P11_2_143.JPG \
    --svm-path results/models/svm_model.pkl
python scripts/cnn_demo.py sample_data/test_data/P11_2_143.JPG \
    --cnn-path results/models/final_cnn_model.pth
```

**Pick one random sample from the test set and run both classifiers on it** (this is
the deliverable for the 30-pt "we should be able to run your programs without any
edits" requirement):

```bash
python scripts/random_test_demo.py
# or: python scripts/random_test_demo.py --seed 42  for a reproducible pick
```

Live webcam demo (uses the trained CNN at ~1 FPS):

```bash
python scripts/cnn_camera.py
```

The presentation video walking through the full pipeline is linked from the top of
`README.md`.

## Part 5 individual contributions

- **Leo:** ran the final 10-epoch CNN training (`results/models/final_cnn_model.pth`)
  and the matching `train_val_cnn_metrics.json`; built the test-set evaluation paths
  in `src/eval.py` (`evaluate_svm_test`, `evaluate_cnn_saved_model_on_test`) including
  the `--svm-test` / `--cnn-test` plumbing; ran both classifiers against the held-out
  test set to produce `results/{svm,cnn}_results/test_*_metrics.json` and confusion
  matrices; built `scripts/cnn_demo.py`, `scripts/svm_demo.py`, and the live-webcam
  `scripts/cnn_camera.py`; recorded and edited the presentation video; cleaned up the
  results directory and trimmed the README to its current form.
- **Alphonsus (Al):** wrote the `scripts/random_test_demo.py` script that picks a
  random sample from `sample_data/test_data/`, runs both the SVM and CNN, and prints
  predictions vs. ground truth (the deliverable for the spec's "pick one random
  sample" requirement); wrote this Part 5 report in `docs/project05_update.md`
  (test-database description, train/val/test metrics table, the why-worse analysis
  with the visual-evidence table and the random-demo seed examples, the six-item
  improvement list, the Note 2 NN-vs-classical comparison, the consolidated run
  instructions); added the link to this report from `README.md`.
