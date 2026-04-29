
# Part 4: Classification (Second Coding Update)

This section covers the classification milestone. We implemented and compared two classifiers
on the hand-gesture dataset: a classical SVM with an RBF kernel trained on hand-crafted
features, and a small CNN trained end-to-end.

## Classifier Justification

### SVM with RBF kernel (primary classifier) (Al)

We chose an RBF-kernel SVM stacked on top of our hand-crafted feature pipeline
(`HOG + contour descriptors + Hu Moments`) for three reasons:

1. **The feature space is high-dimensional but well-behaved.** At 96×96 input with
   16×16 HOG cells we produce ~900 HOG values, plus 6 contour scalars and 7 Hu moments —
   a 913-d feature vector. The classes are 36 letters + digits, and there's no reason to
   expect linear separability in this space, but there's also no reason to expect
   arbitrarily complex boundaries. RBF handles non-linear decision boundaries with a single
   width hyperparameter (`gamma="scale"` scales it automatically to `1/(n_features · var(X))`),
   which is a much milder commitment than choosing e.g. a polynomial degree.
2. **Training data is modest.** 25,200 training images across 36 classes (~700 per class).
   This is a sweet spot for SVM: well within what `libsvm` can train on CPU in minutes,
   but small enough that a from-scratch CNN will hit the memorization floor before the
   generalization floor (see the CNN numbers below).
3. **The features are already invariant.** HOG is translation-tolerant within each cell
   and locally contrast-normalized, and Hu Moments are translation/scale/rotation invariant
   by construction. The SVM doesn't have to re-learn any of that — it only has to learn the
   separating surface. A linear SVM on these features hits roughly 45% val accuracy in our
   earlier notebook runs, so the RBF lift is real.

We wrap the SVM in a `StandardScaler → PCA(n=300) → SVC(C=10, kernel="rbf", gamma="scale")`
`sklearn` pipeline. The PCA step reduces 913-d features to 300-d before the kernel is
applied, which speeds `libsvm` up roughly 3× without hurting accuracy.

### Small custom CNN (comparison) (Leo, Al)

As a second classifier we added a 4-block convolutional network
(`Conv-BN-ReLU-MaxPool` × 4, channel counts 32→64→128→128, global average pool, dropout
0.3, linear head — about 370K parameters). Justification:

1. **Direct comparison to the classical pipeline.** Trained from scratch on exactly the
   same preprocessed images the SVM uses, so the only variable that changes is "features
   from HOG + contours" vs "features learned end-to-end by conv layers".
2. **Cheap to train.** With mixed precision on a Colab T4 (`--amp True`) an epoch takes
   about 30 seconds, and on an M-series Mac via MPS (`pick_device()` prefers CUDA → MPS →
   CPU) it trains in minutes. We deliberately kept it small — a bigger backbone like
   ResNet-50 would overfit even harder on 25k images and 10 subjects.
3. **Satisfies Note 2.** We wanted to show we understood the trade-off between hand-crafted
   features and learned features before the final testing milestone.

## Results

**Subject-wise split** — training uses participants `P1–P7` (25,200 images); validation uses
the held-out participants `P8–P9–P10` (10,800 images). This is the realistic setting: the
model never sees the validation subjects during training.

| Classifier | Train acc. | Val acc. | Val macro-F1 | Val weighted-F1 |
| ---------- | ---------- | -------- | ------------ | --------------- |
| SVM (RBF, PCA-300) — full-image HOG | 1.0000 | **0.7641** | 0.7488 | 0.7488 |
| SVM (RBF, PCA-300) — hand-bbox HOG crop (§ small improvement) | 1.0000 | 0.7475 | 0.7275 | 0.7275 |
| Small CNN (7 epochs) | 0.9904 | 0.5357 | 0.4997 | 0.4997 |

Raw numbers and confusion matrices are saved in:

- `results/svm_metrics.json`, `results/svm_confusion_matrix.npy`,
  `results/svm_confusion_matrix_plot.png`
- `results/cnn_metrics.json`, `results/cnn_confusion_matrix.npy`,
  `results/cnn_confusion_matrix_plot.png`

Both classifiers correctly predict 8,252 (full-image SVM) and 5,786 (CNN) of 10,800
validation images respectively. The bbox-crop SVM predicts 8,073/10,800 correctly. The
per-class breakdown in the confusion matrices shows SVM errors are concentrated on visually
similar sign pairs (`M` vs `N`, `0` vs `O`, `2` vs `V`, `6` vs `W`), which matches the
intuition that those glyphs differ mostly in finger-thumb placement and less in overall
shape.

`results/svm_metrics.json` currently reflects the **bbox-crop** run (the default we ship);
the full-image number is preserved as history in the commit that introduced the Part 4
milestone. Set `hand_bbox_crop=False` in `extract_features` (`src/classifer.py`) to reproduce
the higher-accuracy full-image number.

## Commentary and ideas for improvements

### What the gap tells us

Both classifiers overfit, but to very different degrees:

- **SVM:** train 100.0 % → val 76.4 %  (~24-point gap)
- **CNN:** train 99.0 %  → val 53.6 %  (~45-point gap)

The fact that both hit essentially 100% on train tells us that the classifiers *are capable*
of learning the training distribution; the question is what they're learning. The SVM
generalizes almost twice as well as the CNN. The most plausible explanation is that HOG +
Hu Moments are already designed to be invariant to the things that change between subjects
(translation, scale, illumination, skin tone), so those cues aren't available to the SVM to
overfit on. The CNN, by contrast, is free to latch onto any pixel-level cue that separates
P1–P7 from each other, and many of those cues (skin tone, sleeve colour, lighting) don't
transfer to P8–P10. The existing data augmentation (horizontal flip + brightness jitter) is
not strong enough to wash those cues out.

### Full list of improvements we considered

1. **Crop HOG to the hand bounding box** (implemented in this milestone — see below).
2. **Stronger CNN augmentation.** Rotation ±15°, small translations, random crops, colour
   jitter on H/S/V independently, occlusion/cutout. This is the single cheapest CNN change
   we can make and we expect it to close ~10 points of the train/val gap.
3. **Fine-tune a pretrained backbone.** ResNet-18 pretrained on ImageNet, with the first
   few blocks frozen. The ImageNet prior should make the net more robust to subject-level
   nuisance variation. We didn't do this now because it doubles the training time per epoch
   and we wanted to first see how far we could push the from-scratch CNN.
4. **Subject-normalized preprocessing.** Compute per-subject mean skin tone from the YCrCb
   mask and normalize each subject to the dataset mean. This directly attacks the CNN's
   biggest leak.
5. **Class-balanced sampling / focal loss.** The class distribution is roughly uniform
   (700 images per class), but per-subject counts are not, and some classes (`0`, `O`, `W`,
   `M`, `N`) are much harder than others. Up-weighting them should help macro-F1.
6. **Test-time augmentation.** At eval time, average predictions over horizontal flip and
   ±10° rotation — cheap and almost always worth a point or two.
7. **A richer scalar feature set.** The current contour descriptors are 6-d; we could add
   Fourier descriptors of the contour, inner/outer finger-length ratios from skeletonization,
   and the raw Hu Moments of the convex-hull-minus-hand region. This would roughly double
   the classical feature size without much cost.

### The one small improvement we implemented before final testing

**Hand-bounding-box crop before HOG.** Our segmentation pipeline already produces a clean
single-hand contour via YCrCb skin thresholding + morphology + largest-contour selection,
but the original HOG descriptor was computed over the full preprocessed image, so gradients
from the forearm, shoulder, and (sometimes) face were entering the feature vector. Our
hypothesis going in was that different subjects have very different amounts of visible
"non-hand skin" at the edges of the frame, which would let the SVM latch onto subject
identity rather than sign identity — and that cropping to the hand would close some of the
24-point train/val gap.

**How it's implemented.** The code lives in
`src/pipeline.py:FeatureExtraction.crop_to_contour` and is plumbed through
`extract_all(..., hand_bbox_crop=True)`. For each image we take the segmentation contour,
compute its axis-aligned bounding box, pad by 10 % of the box's width and height (so we
don't clip fingertips), crop the preprocessed image to that box, then resize the crop back
to the HOG input size (96×96) so the descriptor length stays constant at 900-d. If
segmentation fails (empty contour, or bbox smaller than 32 px) we fall back to the full
image. `extract_features` in `src/classifer.py` turns it on by default; the flag is
configurable so both variants can be reproduced.

**What actually happened.** We re-trained and re-evaluated the SVM with the bbox crop on
(`python -m src.train --data data/all --svm True` then the matching `eval`) and the val
accuracy *dropped* from 0.7641 to **0.7475** (−1.66 points absolute, macro-F1 0.7488 →
0.7275). Train accuracy stayed at 1.0000, as expected — the RBF SVM has more than enough
capacity to memorize either feature space.

This is a negative result, and a useful one. The most likely explanation is the
resize-back-to-96×96 step: many hands in the dataset occupy maybe half the frame, so the
padded bounding box is ~60×60 pixels. Stretching that back to 96×96 is a ~1.6× upscale with
`INTER_AREA`, which *blurs the gradients HOG depends on*. The "noise" we were trying to
remove from the background was apparently less harmful than the gradient smearing we
introduced by upsampling. The padding choice of 10 % is likely also too aggressive — on
small crops, 10 % of the bbox is only 5–6 pixels, so any finger that extends diagonally out
of the frame gets clipped.

**What we'd do differently for the final milestone.** Three options, in order of
cheapness:

1. *Don't resize the crop back up.* Instead, letterbox the crop into a fixed 96×96 canvas
   (center the crop, zero-pad to square). HOG's cell grid stays at the native resolution of
   the hand, and the descriptor length changes from 900 to... still 900, because HOG is
   computed over the full padded canvas. No information loss beyond the crop itself.
2. *Use the crop at a larger input size.* Preprocess to 224×224 first, compute HOG at 16×16
   cell size directly on the crop in its native resolution (~120 px), then pad the HOG
   vector to a fixed length with the hand crop's aspect ratio encoded separately.
3. *Keep the full-image HOG, but add a hand-masked branch.* Run the full-image HOG *and*
   a second HOG only over the segmented hand pixels (everything outside the mask zeroed
   out). Concatenate. This gives the SVM both the global context *and* the hand-only signal
   and lets it decide which is useful.

For submission we're shipping the code with `hand_bbox_crop=True` as the documented
Part 4 change, and the Results table above reports both variants so the effect of the
improvement is unambiguous.

## How to run

One-time setup:

```bash
pip install -r requirements.txt
bash scripts/download_dataset.sh        # pulls the ~1.1 GB dataset into data/
```

Train + evaluate the **SVM** (the recommended path — fast on CPU, no GPU needed):

```bash
python -m src.train --data data/ --svm True
python -m src.eval  --data data/ --svm True --svm-path results/svm_model.pkl
```

Train + evaluate the **CNN**. `--amp` turns on mixed precision when CUDA is available
(no-op on CPU/MPS). `--num-workers 2` keeps Colab happy; bump to 4+ on a local Mac:

```bash
python -m src.train --data data/all --epochs 10 --batch-size 64 --num-workers 2 --amp True
python -m src.eval  --data data/all --cnn-path results/checkpoint_epoch_7.pth
```

Run the end-to-end pipeline on a single sample image (a handful are checked in under
`sample_data/`):

```bash
python -m scripts.demo sample_data/P1_B_52.jpg --svm-path results/svm_model.pkl
```

The demo prints the HOG length, the contour descriptors, and — if the SVM model is present —
the predicted class letter.

### Performance notes (locally / on Colab)

- `Preprocessor.preprocess_final` is the hot-path preprocessor used by both the SVM feature
  extractor and the CNN dataloader. It only runs `resize → CLAHE → gaussian blur` and skips
  the `bilateral` + `gamma` variants that `Preprocessor.preprocess` computes but that the
  classifiers don't consume. Bilateral filtering is by far the slowest op in the pipeline,
  so skipping it roughly halves per-image CPU cost.
- The CNN dataloader uses `num_workers=2` + `pin_memory=True` when CUDA is available; the
  training loop uses `torch.cuda.amp.autocast` + `GradScaler` for ~1.5–2× speedup on a
  Colab T4. Device selection (`src/classifer.py:pick_device`) prefers CUDA → Apple MPS →
  CPU so the same code runs fast on Colab and on a Mac without changes.
- SVM training/eval spends most of its time in feature extraction, so re-runs benefit from
  caching the extracted feature matrices. Consider adding `--cache-dir` to `train.py` /
  `eval.py` for the final milestone; for Part 4 we kept the pipeline simple.

## Part 4 individual contributions

- **Leo:** all of the initial Part 4 work — SVM classifier design (RBF kernel, PCA-300,
  `StandardScaler` pipeline) and the `SVMClassifer` wrapper in `src/classifer.py`; CNN
  architecture (`src/classifer.py:CNN`) and the CNN training loop (`train_cnn`); the
  subject-wise dataloader and augmentation logic in `src/dataloader.py`; the
  `extract_features` HOG + contour + Hu feature-vector builder; the first version of the
  train and eval CLIs (`src/train.py`, `src/eval.py`), including the SVM and CNN evaluation
  reporting paths and the confusion-matrix plotting; the initial CNN training run
  (`results/CNN_output_results.md`) and the checkpoint + metrics JSON artifacts it produced.
- **Alphonsus (Al):** everything added in this milestone on top of Leo's Part 4 work —
  the `Preprocessor.preprocess_final` hot-path optimization in `src/pipeline.py`; the
  hand-bounding-box HOG crop (`FeatureExtraction.crop_to_contour` + the `hand_bbox_crop`
  flag on `extract_all`), including the re-run that produced the current
  `results/svm_metrics.json` and confusion-matrix plot; AMP + `pick_device` (CUDA → MPS →
  CPU) support in `train_cnn`; the `--num-workers`, `--amp`, `pin_memory`, and
  `persistent_workers` plumbing in `src/train.py`; bug fixes in `src/eval.py` (CNN plot
  title, printed metrics path); the `scripts/demo.py` single-image runner; `.gitignore`
  updates (PDFs, `*.pth`, cache dirs); and this Part 4 writeup (classifier justification,
  results table, commentary on the bbox-crop regression, follow-up proposals, and run
  instructions).