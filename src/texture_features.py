"""
Texture features using Local Binary Patterns (LBP) applied inside the lesion mask.
Returns histogram bins as individual features for use in a feature vector.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


N_BINS = 16   # number of LBP histogram bins kept as features
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS


def get_texture_features(image, mask):
    """
    Compute LBP texture histogram inside the masked lesion region.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), float32 [0,1] or uint8
    mask  : np.ndarray (H, W)

    Returns
    -------
    dict with keys lbp_0 … lbp_{N_BINS-1} (normalised histogram bins)
    Returns np.nan for all bins if mask is empty.
    """
    nan_result = {f"lbp_{i}": np.nan for i in range(N_BINS)}

    if mask is None or mask.sum() == 0:
        return nan_result

    if image.dtype != np.uint8:
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")

    binary_mask = (mask > 0)
    lbp_values = lbp[binary_mask]

    n_patterns = LBP_N_POINTS + 2  # uniform LBP patterns
    hist, _ = np.histogram(lbp_values, bins=N_BINS, range=(0, n_patterns))
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total

    return {f"lbp_{i}": float(hist[i]) for i in range(N_BINS)}
