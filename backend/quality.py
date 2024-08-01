import pandas as pd
import os
import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def adjust_contrast(image, factor):
    mean = np.mean(image)
    return np.uint8(np.clip((image - mean) * factor + mean, 0, 255))

def adjust_brightness(image, factor):
    image = image.astype(np.float32)
    image = image * factor
    return np.clip(image, 0, 255).astype(np.uint8)

def is_low_contrast(image, threshold=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid_pixels = gray[(gray > 20) & (gray < 255)]
    Q1 = np.percentile(valid_pixels, 5)
    Q3 = np.percentile(valid_pixels, 95)

    IQR = Q3 - Q1
    contrast = IQR / (Q3 + Q1)

    return contrast < threshold

def is_underexposed(image, threshold=70):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid_pixels = gray[(gray > 0) & (gray < 255)]
    hist = cv2.calcHist([valid_pixels], [0], None, [256], [0, 256])
    total_valid_pixels = valid_pixels.size
    cumulative_hist = np.cumsum(hist) / total_valid_pixels
    underexposed_idx = np.argmax(cumulative_hist > 0.5)
    is_underexposed = (underexposed_idx < threshold)

    return is_underexposed

def is_overexposed(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid_pixels = gray[(gray > 0) & (gray < 255)]
    hist = cv2.calcHist([valid_pixels], [0], None, [256], [0, 256])
    total_valid_pixels = valid_pixels.size
    cumulative_hist = np.cumsum(hist) / total_valid_pixels
    overexposed_idx = np.argmax(cumulative_hist > 0.8)

    is_overexposed = (overexposed_idx > threshold)

    return is_overexposed

def evaluate_image_quality(image):
    low_quality = 0

    if is_low_contrast(image, 0.2):
        low_quality = 1
    if is_underexposed(image, threshold=60):
        low_quality = 2
    if is_overexposed(image, 235):
        low_quality = 3

    return low_quality


def aug_and_eval(image_path, multiline_string):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if random.choice([True, False]):
        brightness_delta = np.random.uniform(0.3, 2.0)
        modified_image = adjust_brightness(image, brightness_delta)
    else:
        contrast_factor = np.random.uniform(0.1, 1.5)
        modified_image = adjust_contrast(image, contrast_factor)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    low_quality = evaluate_image_quality(image)
    if not low_quality:
        print("Good Quality")

    plt.imshow(modified_image)
    plt.axis('off')
    plt.show()

    low_quality = evaluate_image_quality(modified_image)
    if low_quality == 0:
        print("Good Quality")
    elif low_quality == 1:
        print("Low Contrast")
    elif low_quality == 2:
        print("Underexposed")
    elif low_quality == 3:
        print("Overexposed")

# df = pd.read_csv(input_csv)
# df_shuffled = df.sample(frac=1)
#
# for index in range(show_number):
#   row = df_shuffled.iloc[index]
#   aug_and_eval(row['path'], row['label'])