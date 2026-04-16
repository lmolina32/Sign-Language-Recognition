#!/usr/bin/env python3

import os
import cv2
import numpy as np
from skimage import feature as skfeature, exposure as skexposure


class Preprocessor:
    def __init__(self):
        pass

    def resize(self, img, target_size: tuple = (224, 224)):
        """Resize to target_size preserving aspect ratio; pad remainder with black."""
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    def gamma_correction(self, img: np.ndarray, gamma: int = 1.0):
        """Perform Gamma correction to image to adjust brightness"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            np.uint8
        )
        return cv2.LUT(img, table)

    def bilateral_filter(
        self, img: np.ndarray, d: int = 9, sigmaColor: int = 75, sigmaSpace=75
    ):
        """Perform bilateral filtering on image to smooth image while preserving edges"""
        return cv2.bilateralFilter(
            img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
        )

    def clahe_enhancement(
        self, img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """CLAHE on the L channel of LAB color space for illumination normalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def gaussian_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Gaussian blur to suppress pixel noise before gradient-based feature extraction."""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Full preprocessing chain: returns every intermediate for debugging/notebooks."""
        gamma_correction = self.gamma_correction(img)
        bilateral_filter = self.bilateral_filter(img)
        return {
            "resize": self.resize(img),
            "gaussian_blur": self.gaussian_blur(img),
            "clahe_enhancement": self.clahe_enhancement(img),
            "gamma_correction": gamma_correction,
            "bilateral_filter": bilateral_filter,
            "final": self.gaussian_blur(self.clahe_enhancement(self.resize(img))),
        }

    def preprocess_final(
        self, img: np.ndarray, target_size: tuple = (224, 224)
    ) -> np.ndarray:
        """Hot-path preprocessing used in training/eval: resize -> CLAHE -> Gaussian blur.

        Skips the gamma/bilateral variants that `preprocess` computes but that the
        downstream models don't consume. Shaves roughly half the per-image CPU cost in
        the SVM feature-extraction loop and the CNN dataloader.
        """
        return self.gaussian_blur(self.clahe_enhancement(self.resize(img, target_size)))

    @staticmethod
    def to_rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_ycrcb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    @staticmethod
    def to_hsv(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


class Segmentation:
    # Empirical skin color range in YCrCb (Kovac et al., 2003)
    LOWER_SKIN = np.array([0, 133, 77], dtype=np.uint8)
    UPPER_SKIN = np.array([255, 173, 127], dtype=np.uint8)

    def __init__(self):
        self.lower_skin = self.LOWER_SKIN.copy()
        self.upper_skin = self.UPPER_SKIN.copy()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def hsv_mask(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 170, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def refine_mask(self, mask: np.ndarray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def ycrcb_mask(self, img: np.ndarray) -> np.ndarray:
        """Binary skin mask via YCrCb thresholding + morphological opening."""
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_skin, self.upper_skin)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        return mask

    def get_largest_contour(self, mask: np.ndarray) -> tuple:
        """Return (contour, clean_mask) keeping only the largest connected region."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, np.zeros_like(mask)
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
        return largest, clean

    def segment(self, img: np.ndarray) -> tuple:
        """Full segmentation pipeline. Returns results, dictionary with all segmentation operations."""
        mask = self.ycrcb_mask(img)
        contour, clean_mask = self.get_largest_contour(mask)
        hsv_mask = self.hsv_mask(img)
        refined_mask = self.refine_mask(hsv_mask)
        return {
            "ycrcb_mask": mask,
            "contour_mask": clean_mask,
            "hsv_mask": hsv_mask,
            "refined_mask": refined_mask,
            "contour": contour,
        }


class FeatureExtraction:

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract_canny_edges(self, img, val: int = 35, ksize: int = 3, ratio: int = 3):
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        low_threshold = val
        img_blur = cv2.blur(src_gray, (3, 3))
        detected_edges = cv2.Canny(
            img_blur, low_threshold, low_threshold * ratio, ksize
        )
        mask = detected_edges != 0
        return img * (mask[:, :, None].astype(img.dtype))

    def extract_hog(self, img: np.ndarray) -> tuple:
        """HOG features + rescaled visualization image.

        Returns:
            feature_vector: 1-D np.ndarray (26244 elements at default settings on 224x224)
            hog_vis: visualization image (same spatial size as input)
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        feats, hog_img = skfeature.hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=True,
            feature_vector=True,
        )
        hog_vis = skexposure.rescale_intensity(hog_img, in_range=(0, 10))
        return feats, hog_vis

    def crop_to_contour(
        self,
        img: np.ndarray,
        contour: np.ndarray,
        pad: float = 0.1,
        min_size: int = 32,
    ) -> np.ndarray:
        """Crop `img` to the contour's bounding box with fractional padding.

        Returns the crop resized back to `img`'s original (H, W) so HOG produces a
        fixed-length descriptor. Falls back to the full image if the contour is
        missing / too small to be useful (e.g. segmentation failure).
        """
        if contour is None or len(contour) < 3:
            return img
        h, w = img.shape[:2]
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_size or bh < min_size:
            return img
        px = int(bw * pad)
        py = int(bh * pad)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(w, x + bw + px)
        y1 = min(h, y + bh + py)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return img
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

    def extract_contour_features(self, contour: np.ndarray, mask: np.ndarray) -> dict:
        """Shape descriptors from the segmented hand contour.

        Returns a dict with: area, perimeter, solidity, aspect_ratio, extent, num_defects
        """
        if contour is None or len(contour) < 3:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "solidity": 0.0,
                "aspect_ratio": 0.0,
                "extent": 0.0,
                "num_defects": 0,
            }
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        extent = area / (w * h) if (w * h) > 0 else 0.0
        num_defects = 0
        try:
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                num_defects = sum(1 for d in defects if d[0][3] / 256.0 > 5)
        except cv2.error:
            pass
        return {
            "area": area,
            "perimeter": perimeter,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "num_defects": num_defects,
        }

    def extract_hu_moments(self, contour_or_mask: np.ndarray) -> np.ndarray:
        """7 log-transformed Hu Moments (invariant to translation, scale, rotation)."""
        if contour_or_mask is None or (
            hasattr(contour_or_mask, "__len__") and len(contour_or_mask) < 3
        ):
            return np.zeros(7, dtype=np.float64)
        M = cv2.moments(contour_or_mask)
        hu = cv2.HuMoments(M).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    def extract_all(
        self,
        img: np.ndarray,
        contour: np.ndarray,
        mask: np.ndarray,
        hand_bbox_crop: bool = False,
    ) -> np.ndarray:
        """Concatenate all features into a single flat vector.

        At default HOG settings on 224x224:
            HOG (26244) + contour descriptors (6) + Hu moments (7) = 26257

        When `hand_bbox_crop=True` the image is cropped to the segmentation contour's
        bounding box (padded 10%) before HOG is computed — HOG descriptor length is
        unchanged because the crop is resized back to the input size. This keeps
        background gradients (face, forearm) out of the feature vector.
        """
        canny_edges = self.extract_canny_edges(img)
        hog_input = (
            self.crop_to_contour(img, contour) if hand_bbox_crop else img
        )
        hog_feats, hog_img = self.extract_hog(hog_input)
        contour_dict = self.extract_contour_features(contour, mask)
        contour_arr = np.array(list(contour_dict.values()), dtype=np.float64)
        hu = self.extract_hu_moments(contour)
        features = np.concatenate([hog_feats, contour_arr, hu])
        return {
            "canny_edges": canny_edges,
            "hog_feats": hog_feats,
            "hog_img": hog_img,
            "contour_dict": contour_dict,
            "contour_arr": contour_arr,
            "hu": hu,
            "features": features,
        }
