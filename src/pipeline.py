#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import feature as skfeature, exposure as skexposure


class Preprocessor:
    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size: tuple = target_size

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize to target_size preserving aspect ratio; pad remainder with black."""
        h, w = img.shape[:2]
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        y_off = (self.target_size[0] - new_h) // 2
        x_off = (self.target_size[1] - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def apply_clahe(self, img: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """CLAHE on the L channel of LAB color space for illumination normalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Gaussian blur to suppress pixel noise before gradient-based feature extraction."""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Full preprocessing chain: resize -> CLAHE -> Gaussian blur."""
        return self.blur(self.apply_clahe(self.resize(img)))

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

    def get_skin_mask(self, img: np.ndarray) -> np.ndarray:
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
        """Full segmentation pipeline. Returns (raw_mask, contour, clean_mask)."""
        mask = self.get_skin_mask(img)
        contour, clean_mask = self.get_largest_contour(mask)
        return mask, contour, clean_mask


class FeatureExtraction:
    def __init__(self, orientations: int = 9, pixels_per_cell: tuple = (8, 8),
                 cells_per_block: tuple = (2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

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

    def extract_contour_features(self, contour: np.ndarray, mask: np.ndarray) -> dict:
        """Shape descriptors from the segmented hand contour.

        Returns a dict with: area, perimeter, solidity, aspect_ratio, extent, num_defects
        """
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
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'num_defects': num_defects,
        }

    def extract_hu_moments(self, contour_or_mask: np.ndarray) -> np.ndarray:
        """7 log-transformed Hu Moments (invariant to translation, scale, rotation)."""
        M = cv2.moments(contour_or_mask)
        hu = cv2.HuMoments(M).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    def extract_all(self, img: np.ndarray, contour: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        """Concatenate all features into a single flat vector.

        At default HOG settings on 224x224:
            HOG (26244) + contour descriptors (6) + Hu moments (7) = 26257
        """
        hog_feats, _ = self.extract_hog(img)
        contour_dict = self.extract_contour_features(contour, mask)
        contour_arr = np.array(list(contour_dict.values()), dtype=np.float64)
        hu = self.extract_hu_moments(contour)
        return np.concatenate([hog_feats, contour_arr, hu])
