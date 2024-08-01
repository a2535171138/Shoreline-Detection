import numpy as np
import cv2
import logging


def calculate_confidence(predicted_img):
    hist, bin_edges = np.histogram(predicted_img, bins=256, range=(0, 255), density=True)
    hist_norm = hist / hist.sum()

    gray_levels = np.arange(256)
    mean = np.sum(hist_norm * gray_levels)
    std_dev = np.sqrt(np.sum(hist_norm * (gray_levels - mean) ** 2))

    max_std_dev = np.sqrt(256 ** 2 / 12)
    confidence = np.sqrt(1 - (std_dev / max_std_dev))


    return confidence



