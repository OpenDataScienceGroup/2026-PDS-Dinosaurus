"""
Color features extracted from the lesion region defined by the mask.
Features: mean/std of R, G, B channels; mean/std of H, S, V channels; color entropy.
"""

import numpy as np
import cv2


def get_color_features(image, mask):
    """
    Extract color statistics from the lesion region.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3), float32 in [0,1] or uint8 in [0,255]
    mask  : np.ndarray  (H, W), any dtype – non-zero pixels are lesion

    Returns
    -------
    dict with keys:
        mean_r, mean_g, mean_b, std_r, std_g, std_b
        mean_h, mean_s, mean_v, std_h, std_s, std_v
        color_entropy
    Returns np.nan for every feature if mask is empty.
    """
    if mask is None or mask.sum() == 0:
        nan = np.nan
        return {k: nan for k in [
            "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
            "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
            "color_entropy",
        ]}

    binary_mask = (mask > 0)

    # ensure uint8 for OpenCV
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    # --- RGB features ---
    r = img_uint8[:, :, 0][binary_mask].astype(float)
    g = img_uint8[:, :, 1][binary_mask].astype(float)
    b = img_uint8[:, :, 2][binary_mask].astype(float)

    # --- HSV features ---
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][binary_mask].astype(float)
    s = hsv[:, :, 1][binary_mask].astype(float)
    v = hsv[:, :, 2][binary_mask].astype(float)

    # --- Color entropy (on grayscale inside mask) ---
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)[binary_mask]
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))

    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
        "mean_h": float(h.mean()),
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "std_h": float(h.std()),
        "std_s": float(s.std()),
        "std_v": float(v.std()),
        "color_entropy": float(entropy),
    }
