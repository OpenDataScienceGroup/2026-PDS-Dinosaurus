"""
Color features extracted from the lesion region defined by the mask.

Features:
  - Mean and std of R, G, B channels
  - Mean and std of H, S, V channels
  - Color entropy (grayscale histogram entropy inside mask)
  - Blue-white veil score (fraction of lesion pixels where blue > red —
    a clinically recognised indicator of melanoma)
  - Melanoma color count (number of the six canonical melanoma-associated
    colours present in the lesion, based on Kasmi & Mokrani 2016)
"""

import numpy as np
import cv2

# Six canonical melanoma-associated colours in HSV (OpenCV scale: H 0-180, S 0-255, V 0-255)
# Based on Kasmi & Mokrani (2016): white, black, red, light brown, dark brown, blue-grey
MELANOMA_COLORS_HSV = {
    "white":       ([0,   0,   200], [180, 30,  255]),
    "black":       ([0,   0,   0],   [180, 255, 50 ]),
    "red":         ([0,   100, 50],  [10,  255, 255]),
    "light_brown": ([10,  50,  100], [20,  200, 200]),
    "dark_brown":  ([5,   100, 50],  [15,  255, 150]),
    "blue_grey":   ([90,  20,  50],  [130, 150, 200]),
}

# Minimum number of lesion pixels that must match a colour for it to count
_MIN_COLOR_PIXELS = 10


def get_color_features(image, mask):
    """
    Extract color statistics from the lesion region.

    Combines per-channel RGB/HSV statistics with clinically motivated
    features: blue-white veil and melanoma colour count.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3), float32 in [0,1] or uint8 in [0,255]
            Expected in RGB channel order.
    mask  : np.ndarray  (H, W), any dtype - non-zero pixels are lesion

    Returns
    -------
    dict with keys:
        mean_r, mean_g, mean_b         - mean RGB per channel
        std_r,  std_g,  std_b          - std RGB per channel
        mean_h, mean_s, mean_v         - mean HSV per channel
        std_h,  std_s,  std_v          - std HSV per channel
        color_entropy                  - grayscale histogram entropy
        blue_veil_score                - fraction of lesion pixels where blue > red
        melanoma_color_count           - number of canonical melanoma colours present
                                         (0-6, based on Kasmi & Mokrani 2016)
    Returns np.nan for every feature if mask is empty.
    """
    nan_keys = [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
        "color_entropy", "blue_veil_score", "melanoma_color_count",
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
    # low std indicates uniform colour - likely benign
    # high std indicates variable colour - more likely malignant

    # --- HSV features ---
    # HSV separates colour from brightness, making it more robust to lighting variation
    # image is RGB so use COLOR_RGB2HSV (not BGR2HSV)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][binary_mask].astype(float)  # hue: the colour itself
    s = hsv[:, :, 1][binary_mask].astype(float)  # saturation: colour intensity
    v = hsv[:, :, 2][binary_mask].astype(float)  # value: brightness
    # cancer lesions are often unevenly pigmented - high saturation variance
    # can be a good indicator of malignancy

    # --- Color entropy (grayscale histogram inside mask) ---
    # low entropy = uniform texture, high entropy = complex/varied texture
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)[binary_mask]
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))

    # --- Blue-white veil score ---
    # Fraction of lesion pixels where blue channel exceeds red channel.
    # Blue-white veil is a recognised dermoscopy criterion for melanoma -
    # a whitish-blue haze over the lesion caused by melanin in the dermis.
    # In RGB: channel 0 = red, channel 2 = blue (b > r approximates the veil)
    blue_veil_score = float(np.sum(b > r) / len(r)) if len(r) > 0 else np.nan

    # --- Melanoma colour count ---
    # Count how many of the six canonical melanoma-associated colours are
    # present in the lesion. More distinct colours correlate with higher
    # malignancy risk (Kasmi & Mokrani 2016).
    melanoma_color_count = _count_melanoma_colors(hsv, binary_mask)

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
        "melanoma_color_count": melanoma_color_count,
    }


def _count_melanoma_colors(hsv_image, binary_mask):
    """
    Count how many of the six canonical melanoma colours are present
    in the lesion region.

    Parameters
    ----------
    hsv_image   : np.ndarray (H, W, 3) uint8, already converted to HSV
    binary_mask : np.ndarray (H, W) bool

    Returns
    -------
    int - number of colours found (0 to 6)
    """
    lesion_hsv = hsv_image[binary_mask]
    count = 0
    for name, (lower, upper) in MELANOMA_COLORS_HSV.items():
        lower = np.array(lower)
        upper = np.array(upper)
        in_range = np.all((lesion_hsv >= lower) & (lesion_hsv <= upper), axis=1)
        if in_range.sum() >= _MIN_COLOR_PIXELS:
            count += 1
    return count
