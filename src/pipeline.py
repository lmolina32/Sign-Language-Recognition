#!/usr/bin/env python3

import cv2 
import numpy as np 
import os 

class Preprocessor:
    def __init__(self):
        pass

    def resize(self, img, target_size: tuple=(224,224)):
        """resize the img to the target size while keeping ratio"""
        h, w = img.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        y_off = (target_size[0] - new_h) // 2
        x_off = (target_size[1] - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def gamma_correction(self, img, gamma: int=1.0):
        """Perform Gamma correction to image to adjust brightness"""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(img, table)

    def gaussian_blur(self, img, ksize: tuple=(5,5), sigma: float=0.0):
        """Perform gaussian blur on the img for smoothing"""
        return cv2.GaussianBlur(img, ksize, sigma)

    def clahe_enhancement(self, image, clip_limit=2.0, tile_size=8):
        """Perform clahe enchancement for improving local contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def bilateral_filter(self, img, d: int=9, sigmaColor: int=75, sigmaSpace=75):
        """Perform bilateral filtering on image to smooth image while preserving edges"""
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    def run(self, img): 
        pass 
    
class Segmentation:
    def __init__(self):
        pass
    
class FeatureExtraction():
    def __init__(self):
        pass