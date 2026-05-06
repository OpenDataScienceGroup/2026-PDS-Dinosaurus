"""
Color features extracted from the lesion region defined by the mask.

Features:
  - Mean and std of R, G, B channels
  - Mean and std of H, S, V channels
  - Color entropy (grayscale histogram entropy inside mask)
  - Blue-white veil score (fraction of lesion pixels where blue > red —
    a clinically recognised indicator of melanoma)
  - Dominant color count via K-means (number of visually distinct colors
    in the lesion, reflecting the dermoscopy color-count criterion)
"""

import numpy as np
import cv2

# Number of clusters for K-means color counting
_KMEANS_K = 3
_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_KMEANS_ATTEMPTS = 10


def get_color_features(image, mask):
    """
    Extract color statistics from the lesion region.

    Combines per-channel RGB/HSV statistics with clinically motivated
    features: blue-white veil and dominant color count.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3), float32 in [0,1] or uint8 in [0,255]
            Expected in RGB channel order.
    mask  : np.ndarray  (H, W), any dtype – non-zero pixels are lesion

    Returns
    -------
    dict with keys:
        mean_r, mean_g, mean_b         – mean RGB per channel
        std_r,  std_g,  std_b          – std RGB per channel
        mean_h, mean_s, mean_v         – mean HSV per channel
        std_h,  std_s,  std_v          – std HSV per channel
        color_entropy                  – grayscale histogram entropy
        blue_veil_score                – fraction of lesion pixels where blue > red
                                         (approximates blue-white veil)
        dominant_color_count           – number of distinct K-means color clusters
                                         found in the lesion (k=3)
    Returns np.nan for every feature if mask is empty.
    """
    nan_keys = [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
        "color_entropy", "blue_veil_score", "dominant_color_count",
    ]

    if mask is None or mask.sum() == 0:
        return {k: np.nan for k in nan_keys}

    binary_mask = (mask > 0)

    # ensure uint8 for OpenCV (pipeline passes RGB uint8, but guard for float input)
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    # --- RGB features ---
    # channel 0 = red, 1 = green, 2 = blue in RGB order
    r = img_uint8[:, :, 0][binary_mask].astype(float)
    g = img_uint8[:, :, 1][binary_mask].astype(float)
    b = img_uint8[:, :, 2][binary_mask].astype(float)
    # low std indicates uniform colour — likely benign
    # high std indicates variable colour — more likely malignant

    # --- HSV features ---
    # HSV separates colour from brightness, making it more robust to lighting variation
    # image is RGB so use COLOR_RGB2HSV (not BGR2HSV)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][binary_mask].astype(float)  # hue: the colour itself
    s = hsv[:, :, 1][binary_mask].astype(float)  # saturation: colour intensity
    v = hsv[:, :, 2][binary_mask].astype(float)  # value: brightness
    # cancer lesions are often unevenly pigmented — high saturation variance
    # can be a good indicator of malignancy

    # --- Color entropy (grayscale histogram inside mask) ---
    # low entropy = uniform texture, high entropy = complex/varied texture
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)[binary_mask]
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))

    # --- Blue-white veil score ---
    # Fraction of lesion pixels where blue channel exceeds red channel.
    # Blue-white veil is a recognised dermoscopy criterion for melanoma —
    # a whitish-blue haze over the lesion caused by melanin in the dermis.
    # In RGB: channel 0 = red, channel 2 = blue (b > r approximates the veil)
    blue_veil_score = float(np.sum(b > r) / len(r)) if len(r) > 0 else np.nan

    # --- Dominant color count via K-means ---
    # Groups lesion pixels into k=3 colour clusters and counts how many
    # are actually populated. Reflects the dermoscopy ABCD colour-count
    # criterion — more distinct colours correlate with higher malignancy risk.
    dominant_color_count = _count_dominant_colors(img_uint8, binary_mask)

    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r":  float(r.std()),
        "std_g":  float(g.std()),
        "std_b":  float(b.std()),
        "mean_h": float(h.mean()),
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "std_h":  float(h.std()),
        "std_s":  float(s.std()),
        "std_v":  float(v.std()),
        "color_entropy":        float(entropy),
        "blue_veil_score":      blue_veil_score,
        "dominant_color_count": dominant_color_count,
    }


def _count_dominant_colors(img_uint8, binary_mask):
    """
    Run K-means on lesion pixels and return the number of populated clusters.

    Parameters
    ----------
    img_uint8   : np.ndarray (H, W, 3) uint8, RGB
    binary_mask : np.ndarray (H, W) bool

    Returns
    -------
    int – number of unique clusters found (≤ _KMEANS_K), or np.nan on failure
    """
    pixels = img_uint8[binary_mask].astype(np.float32)

    # need at least k pixels to form k clusters
    if len(pixels) < _KMEANS_K:
        return np.nan

    try:
        _, labels, _ = cv2.kmeans(
            pixels,
            _KMEANS_K,
            None,
            _KMEANS_CRITERIA,
            _KMEANS_ATTEMPTS,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        return int(len(np.unique(labels)))
    except Exception:
        return np.nan
