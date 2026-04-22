"""
Hair detection and removal using BlackHat morphological filtering.
Inspired by Exercise 06 (FYP2026_06_shortcuts.ipynb).

Functions:
  hair_coverage(img_gray)        → float in [0, 1]
  remove_hair(img_rgb, img_gray) → (hair_mask, cleaned_img_rgb)
"""

import numpy as np
import cv2


def hair_coverage(img_gray):
    """
    Estimate the proportion of the image covered by hair using BlackHat filtering.

    Parameters
    ----------
    img_gray : np.ndarray (H, W) uint8 grayscale image

    Returns
    -------
    float in [0, 1] — fraction of pixels identified as hair
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    total_area = img_gray.size
    hair_area = int(hair_mask.sum() / 255)
    return round(hair_area / total_area, 4)


def remove_hair(img_rgb, img_gray, coverage=None):
    """
    Remove hair from an RGB image using BlackHat + inpainting.
    Parameters are automatically adapted based on hair coverage.

    Parameters
    ----------
    img_rgb  : np.ndarray (H, W, 3) uint8
    img_gray : np.ndarray (H, W)    uint8
    coverage : float or None — if None it is computed internally

    Returns
    -------
    hair_mask    : np.ndarray (H, W) uint8 binary hair mask
    cleaned_img  : np.ndarray (H, W, 3) uint8 hair-removed image
    """
    if coverage is None:
        coverage = hair_coverage(img_gray)

    # adapt parameters to hair density
    if coverage < 0.02:          # minimal hair — light touch
        kernel_size, threshold, radius = 5, 15, 3
    elif coverage < 0.06:        # moderate hair
        kernel_size, threshold, radius = 7, 10, 5
    else:                        # heavy hair
        kernel_size, threshold, radius = 11, 8, 7

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    cleaned_img = cv2.inpaint(img_rgb, hair_mask, radius, cv2.INPAINT_TELEA)
    return hair_mask, cleaned_img
