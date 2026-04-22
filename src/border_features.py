"""
Border / shape features extracted from the lesion mask.
Features: area, perimeter, compactness, solidity, eccentricity.
"""

import numpy as np
import cv2


def get_border_features(mask):
    """
    Extract shape descriptors from the lesion mask.

    Parameters
    ----------
    mask : np.ndarray (H, W)

    Returns
    -------
    dict with keys: area, perimeter, compactness, solidity, eccentricity
    Returns np.nan for every feature if mask is empty or no contour found.
    """
    nan_result = {k: np.nan for k in [
        "area", "perimeter", "compactness", "solidity", "eccentricity"
    ]}

    if mask is None or mask.sum() == 0:
        return nan_result

    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return nan_result

    # use the largest contour
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area == 0:
        return nan_result

    perimeter = cv2.arcLength(cnt, True)

    # compactness: perimeter^2 / (4*pi*area)  → 1 for a perfect circle, > 1 otherwise
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.nan

    # solidity: area / convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else np.nan

    # eccentricity from the fitted ellipse (requires >= 5 points)
    eccentricity = np.nan
    if len(cnt) >= 5:
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                eccentricity = float(np.sqrt(1 - (b / a) ** 2))
        except Exception:
            pass

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "compactness": float(compactness) if compactness is not np.nan else np.nan,
        "solidity": float(solidity) if solidity is not np.nan else np.nan,
        "eccentricity": eccentricity,
    }
