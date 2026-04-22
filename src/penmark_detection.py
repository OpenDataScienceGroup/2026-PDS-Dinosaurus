"""
Pen mark detection using a combination of:
  1. Hue-based segmentation targeting blue/purple ink colours.
  2. Morphological filtering to reject hair-like thin structures.

Functions:
  penmark_coverage(img_rgb) → float in [0, 1]
  remove_penmarks(img_rgb)  → (penmark_mask, cleaned_img_rgb)
"""

import numpy as np
import cv2


def penmark_coverage(img_rgb):
    """
    Estimate what fraction of the image is covered by pen marks.

    Pen marks are typically blue, purple, or dark-coloured thin lines
    wider than individual hairs. We detect them in HSV colour space.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) uint8

    Returns
    -------
    float in [0, 1]
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # blue-purple hue range (hue is 0-180 in OpenCV)
    lower_blue = np.array([100, 50, 30])
    upper_blue = np.array([160, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # dark marks (very dark value, any hue) — covers black/dark pens
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    combined = cv2.bitwise_or(mask_blue, mask_dark)

    # morphological opening to remove tiny noise (keep only pen-sized structures)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    total = img_rgb.shape[0] * img_rgb.shape[1]
    pen_area = int(cleaned.sum() / 255)
    return round(pen_area / total, 4)


def remove_penmarks(img_rgb):
    """
    Remove pen marks from an RGB image via inpainting.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) uint8

    Returns
    -------
    penmark_mask : np.ndarray (H, W) uint8
    cleaned_img  : np.ndarray (H, W, 3) uint8
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([100, 50, 30])
    upper_blue = np.array([160, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    combined = cv2.bitwise_or(mask_blue, mask_dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    penmark_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # dilate slightly so inpainting covers edges
    penmark_mask = cv2.dilate(penmark_mask, np.ones((3, 3), np.uint8), iterations=1)

    cleaned_img = cv2.inpaint(img_rgb, penmark_mask, 5, cv2.INPAINT_TELEA)
    return penmark_mask, cleaned_img
