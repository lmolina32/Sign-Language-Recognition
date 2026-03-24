#!/usr/bin/env python3

import cv2 
import numpy as np 
import os 

class Preprocessor:
    def __init__(self, target_size: set=(224, 224)):
        self.target_size: set = target_size

    def resize(self, img, target_size: set=(224,224)):
        """resize the img to the target size while keeping ratio"""
        h, w = img.shape[:2]
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        y_off = (self.target_size[0] - new_h) // 2
        x_off = (self.target_size[1] - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def run(self, img): 
        pass 
    
class Segmentation:
    def __init__(self):
        pass
    
class FeatureExtraction():
    def __init__(self):
        pass