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
├── tests/                   # Testing src code 
```
# Methods Applied and Justification
## Preprocessing 
### Resize (Leo)

### Gamma Correction (Leo)

### Bilateral Filtering (Leo)

### Clahe enhancement (Al)

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the L channel of the LAB color space using a clip limit of 2.0 and an 8×8 tile grid. The idea is that regular histogram equalization works globally — it looks at the whole image and stretches contrast across the board — but that tends to wash things out or over-amplify noise in already well-lit areas. CLAHE fixes this by dividing the image into small tiles and equalizing each one independently, then blending the results so there are no hard seams. Working in LAB means we only touch the luminance (L) and leave the color channels (A, B) alone, so skin tone doesn't get distorted in the process. This matters a lot for our dataset since it's 10 different people, shot in different environments — some images are darker, some have harsher lighting — and we need the hand to look consistent before we try to segment it.

### Gaussian Blur (Leo)

## Segmentation 

### HSV mask (Leo)

### Morphological Operators (`refine_mask`) (Leo)

### YCrCb Mask (Al)

For segmentation, the goal is to isolate the hand from the background. We do this by thresholding on skin color in the YCrCb color space, using the range Cr: [133, 173], Cb: [77, 127] from Kovac et al. (2003). YCrCb is a better choice than RGB or even HSV for this because it separates luminance (Y) from the color information (Cr, Cb). Skin color in Cr-Cb space is surprisingly stable across different lighting conditions — the hue of skin doesn't shift as dramatically when the brightness changes. After thresholding we get a noisy raw binary mask, so we clean it up with morphological operations: erode twice to break thin bridges connecting the hand to other skin-colored regions in the background, then dilate twice to restore the hand's actual boundary. We use an elliptical kernel because hand blobs have curved edges, not rectangular ones.

### Finding Contours (Al)

After the morphological cleanup, there can still be multiple disconnected blobs in the mask — parts of the wrist, arm, or face that made it through the threshold. To get a clean single-hand mask, we find all external contours in the binary image and keep only the largest one by area. That region gets filled and returned as the final clean mask, along with the contour itself for use in feature extraction.

## Feature Extraction 

### Canny Edges (Leo)

### Hog Features (Al)

HOG (Histogram of Oriented Gradients) is our main feature descriptor. The idea is to divide the image into small cells (8×8 pixels), compute the gradient direction and magnitude at each pixel, and bin those directions into a histogram for each cell. We use 9 orientation bins and group cells into 2×2 blocks for local contrast normalization. At 224×224 this gives us a 26,244-element feature vector. The reason HOG works well for sign language is that hand signs are fundamentally about shape — the angles of your fingers, the curve of your palm, whether your hand is open or closed. HOG directly captures the distribution of edge directions in local regions, which maps pretty naturally onto those geometric structures. It's also more robust to lighting changes than raw pixel values since it's based on relative gradient magnitudes.

### Contour Features (Al)

From the segmented hand contour we extract six scalar shape descriptors: area, perimeter, solidity (area / convex hull area), aspect ratio (width / height of the bounding rect), extent (area / bounding rect area), and convexity defect count (number of significant finger gaps, where depth > 5px). These complement HOG by capturing global shape properties that local gradient features don't directly encode. Solidity is particularly useful — a closed fist like "B" or "S" has high solidity (~0.9) while an open hand like "Y" or "G" has a lower value because the fingers create concavities in the convex hull. Defect count is a rough proxy for how many fingers are extended, which is obviously important for distinguishing signs.

### Hu moments (Al)

We also compute the seven Hu Moments from the contour, log-transformed for numerical stability. Hu Moments are derived from image moments and are invariant to translation, scale, and rotation — meaning the same sign held at a slightly different angle or distance should produce similar moment values. This makes the classifier more robust to natural variation in how someone positions their hand. We use the contour directly rather than the full image so we're computing moments on the hand shape only, not the background.

# Results 
## Preprocessing (Leo)

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

# Individual Contributions 
- Leo: 
- Alphonsus: CLAHE enhancement, YCrCb skin segmentation, contour extraction, HOG feature extraction, contour shape descriptors, Hu Moments, segmentation and feature extraction results in the README