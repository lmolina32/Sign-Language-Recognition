# Sign-Language-Recognition Project

**Team:** Leonardo Molina, Alphonsus Koong Bok Hui  
**Course:** CSE 40535 Computer Vision, Spring 2026  

# Repository Structure
```bash
sign-language-recognition/
├── notebooks/               # Python notebooks for testing ideas
├── sample_data/             # Sample Images
├── scripts/                 # Download Data from source 
├── src                      # Download Data from source 
│   └── pipeline.py          # Full pipeline to transform imgs
└── tests/                   # Testing src code 
```
# Methods Applied and Justification
## Preprocessing 
### Resize (Leo)

The resize operation resizes the image to the specified size while maintaining the ratio of the image. This is the first preprocessing step applied to all images as they all varied in size. We mainly applied this transformation to have consistent image dimensions when feeding into our neural network. Laslty, many pretrained models use images that are (224, 244) and thus this is the default size that all images are resized into. This can be easily changed by the passing in `target_size` in the constructor of the `Preprocessor` class. 

### Gamma Correction (Leo)

Gamma correction is a non-linear adjustment of image brightness used to map pixel values on light intensity. In application it prevents images from appearing too dark or washed out. We did this because the dataset was captured in indoor and outdoor with varying lighting. Thus, we applied gamma correction to produce images that are more similar to each other and thus the only difference being the hand gestures. Then when these images are feed into our neural network the features learned are on the actual hand rather than on the differences in brigthness/contrast. Note, I used this [website](https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/) to get the transformation to work. 

### Bilateral Filtering (Leo)

Bilaterla Filtering removes noise from an image while keeping edges sharp. In the dataset we have 10 different participants who took the images in various different lighting and mostly uniform background. This operation was mainly performed to remove any noise that images had but also maintaining the edges assoicated with the hand. We believe this will be important when we pass these images into the Neural Network in order to learn features from the actual hand gesture rather than the noise. Note I just this [website](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) to get my transformation to work. 

### Clahe enhancement (Al)

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the L channel of the LAB color space using a clip limit of 2.0 and an 8×8 tile grid. The idea is that regular histogram equalization works globally — it looks at the whole image and stretches contrast across the board — but that tends to wash things out or over-amplify noise in already well-lit areas. CLAHE fixes this by dividing the image into small tiles and equalizing each one independently, then blending the results so there are no hard seams. Working in LAB means we only touch the luminance (L) and leave the color channels (A, B) alone, so skin tone doesn't get distorted in the process. This matters a lot for our dataset since it's 10 different people, shot in different environments — some images are darker, some have harsher lighting — and we need the hand to look consistent before we try to segment it.

### Gaussian Blur (Leo)

This operation is similar to bilateral filtering as it removes noise from the image by convolving the image with a low-pass filter kernel. The only difference between Gaussian Blur and Bilateral filtering is that Gaussian Blur does not maintain the edges of the images. Nonetheless we still believed this fundamental operation must be part of our pipeline to reduce noise in images. Again the same [website](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) was used. 

## Segmentation 

### HSV mask (Leo)

HSV mask sets a range from the lower and the upper bound of pixels it can capture. This idea was taken from practical 1 with the invisibility cloak applied to our hand segementation. We implemented this to create one way to segment the hand from the background. Our idea was to perform and bitwise and operation with a mask to extract the hand fully from the background. This extracted hand could be used to be feed into our neural netwok. 

### Morphological Operators (`refine_mask`) (Leo)

Morphological operators were used to refine the HSV mask that was created. This is because by itself the map had holes inside it that needed to be cleaned up. The Morphological operators used were `OPEN` and `CLOSE` with an ellipse kernel. [Website](https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html) used and class code. 

### YCrCb Mask (Al)

For segmentation, the goal is to isolate the hand from the background. We do this by thresholding on skin color in the YCrCb color space, using the range Cr: [133, 173], Cb: [77, 127] from Kovac et al. (2003). YCrCb is a better choice than RGB or even HSV for this because it separates luminance (Y) from the color information (Cr, Cb). Skin color in Cr-Cb space is surprisingly stable across different lighting conditions — the hue of skin doesn't shift as dramatically when the brightness changes. After thresholding we get a noisy raw binary mask, so we clean it up with morphological operations: erode twice to break thin bridges connecting the hand to other skin-colored regions in the background, then dilate twice to restore the hand's actual boundary. We use an elliptical kernel because hand blobs have curved edges, not rectangular ones.

### Finding Contours (Al)

After the morphological cleanup, there can still be multiple disconnected blobs in the mask — parts of the wrist, arm, or face that made it through the threshold. To get a clean single-hand mask, we find all external contours in the binary image and keep only the largest one by area. That region gets filled and returned as the final clean mask, along with the contour itself for use in feature extraction.

## Feature Extraction 

### Canny Edges (Leo)

Canny edges aims to satisfy three main criteria: 1. low error rate 2. Good localization 3. Minimal Response. More details are found on this [website](https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html). The main purpose of this feature extraction was to get the edges from the actual hand. Getting the edges from the hand as well as the edges from the fingers allows use to distinguish different hand gestures. Since we are doing neural networks we would like to compare our logits to see if the NN learned anything on the edges and compare the different features. In the notebook exploration I ran the code from the website to find the best `low_threshold` for the images in the dataset. 

### Hog Features (Al)

HOG (Histogram of Oriented Gradients) is our main feature descriptor. The idea is to divide the image into small cells (8×8 pixels), compute the gradient direction and magnitude at each pixel, and bin those directions into a histogram for each cell. We use 9 orientation bins and group cells into 2×2 blocks for local contrast normalization. At 224×224 this gives us a 26,244-element feature vector. The reason HOG works well for sign language is that hand signs are fundamentally about shape — the angles of your fingers, the curve of your palm, whether your hand is open or closed. HOG directly captures the distribution of edge directions in local regions, which maps pretty naturally onto those geometric structures. It's also more robust to lighting changes than raw pixel values since it's based on relative gradient magnitudes.

### Contour Features (Al)

From the segmented hand contour we extract six scalar shape descriptors: area, perimeter, solidity (area / convex hull area), aspect ratio (width / height of the bounding rect), extent (area / bounding rect area), and convexity defect count (number of significant finger gaps, where depth > 5px). These complement HOG by capturing global shape properties that local gradient features don't directly encode. Solidity is particularly useful — a closed fist like "B" or "S" has high solidity (~0.9) while an open hand like "Y" or "G" has a lower value because the fingers create concavities in the convex hull. Defect count is a rough proxy for how many fingers are extended, which is obviously important for distinguishing signs.

### Hu moments (Al)

We also compute the seven Hu Moments from the contour, log-transformed for numerical stability. Hu Moments are derived from image moments and are invariant to translation, scale, and rotation — meaning the same sign held at a slightly different angle or distance should produce similar moment values. This makes the classifier more robust to natural variation in how someone positions their hand. We use the contour directly rather than the full image so we're computing moments on the hand shape only, not the background.

# Results 
## Preprocessing (Leo)
Below are all the preprocessing operations done separately from each other: Resizing, Gaussian Blur, Clahe Enhancement, Gamma correction and Bilateral Filtering 

![Preprocessing Results](sample_data/project03_figures/preprocessing_results.png)

On the left bottom corner we have the final pipeline which consists of resize -> gaussian blur -> Clahe Ehancement. But, since our code is built in a class structure it is really easy to interchange the order of preprocessing. 

## Segmentation (Al)

We compared four segmentation approaches on a set of sample images: YCrCb thresholding, HSV thresholding, Otsu thresholding on grayscale, and GrabCut. The comparison is shown below.

![Segmentation method comparison](sample_data/project03_figures/seg_comparison.png)

YCrCb came out the most consistent across the dataset. HSV thresholding was hit or miss depending on the background color — it struggled whenever the background had warm tones similar to skin. Otsu failed on images where the hand and background had similar overall brightness, which happened fairly often. GrabCut was the slowest and tended to miss finger tips at the edges of the bounding rect.

The YCrCb results across all 15 sample images:

![YCrCb segmentation on all samples](sample_data/project03_figures/seg_results.png)

## Feature Extraction (Al)

**HOG:** The visualization below shows HOG gradient maps for three different signs. You can see distinct patterns corresponding to finger orientations — the edge directions in the HOG image are clearly different between an open hand (Y) and a closed fist (B), which is exactly what we want the classifier to pick up on.

![HOG feature visualization](sample_data/project03_figures/hog_comparison.png)

**Contour features:** The plot below shows the contour, convex hull, bounding rect, and convexity defect points overlaid on the segmented hand. The scalar values (solidity, aspect ratio, defect count) printed in the titles give a sense of how these features differ between signs.

![Contour feature visualization](sample_data/project03_figures/contour_features.png)

**Hu Moments:** The bar chart compares log-transformed Hu Moment values across several sign classes. The pattern of values differs meaningfully between signs, especially in the first two or three moments which capture the most variance in overall shape.

![Hu Moments comparison](sample_data/project03_figures/hu_moments.png)
# How to run the code (Leo)
* More examples on how to run the code in `test/pipeline_test.ipynb`.

```python
import pipeline
import cv2

img = cv2.imread("file_path")
preprocessor = Preprocessor()
segmentation = Segmentation()
feature_extractor = FeatureExtraction()

# All classes return dictionary with images
preprocessor_results = preprocessor.preprocess(img)
segmentation_results = segmentation.segment(img)
feature_results = feature_extractor.extract_all(img, segmentation_results['contour'], segmentation_results['ycrcb_mask'])
```

# Individual Contributions 
- Note: We completed preprocessing, segmentation, and feature extraction independently to get as many ideas as we could. Leos branch is called `feat-preprocessing-leo` Alphonsus branch is called `feature/preprocessing-extraction`. Alphonsus branch was merged into Leos branch and all merge conflicts were resolved then merged into main. 
- Leo: Resize, Gamma correction, Bilateral Filtering, Gaussian Blur, HSV mask, Morphological Operators, Canny Edges, How to run the code, preprocessing, and merging the files togehter. 
- Alphonsus: CLAHE enhancement, YCrCb skin segmentation, contour extraction, HOG feature extraction, contour shape descriptors, Hu Moments, segmentation and feature extraction results in the README

# Part 4: Classification (Second Coding Update)

This section covers the classification milestone. We implemented and compared two classifiers
on the hand-gesture dataset: a classical SVM with an RBF kernel trained on hand-crafted
features, and a small CNN trained end-to-end (Note 2 of the spec — we're allowed to include
NNs now, so we include one and compare).

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
bash scripts/download_dataset.sh        # pulls the ~1.1 GB dataset into data/all
```

Train + evaluate the **SVM** (the recommended path — fast on CPU, no GPU needed):

```bash
python -m src.train --data data/all --svm True
python -m src.eval  --data data/all --svm True --svm-path results/svm_model.pkl
```

Train + evaluate the **CNN**. `--amp` turns on mixed precision when CUDA is available
(no-op on CPU/MPS). `--num-workers 2` keeps Colab happy; bump to 4+ on a local Mac:

```bash
python -m src.train --data data/all --epochs 10 --batch-size 64 --num-workers 2 --amp True
python -m src.eval  --data data/all --cnn-path results/checkpoint_epoch_10.pth
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