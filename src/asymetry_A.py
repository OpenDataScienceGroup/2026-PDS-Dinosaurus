"""
Feature A: Asymmetry extraction - rotating around the middle
Assymmetry = area of difference / total area
"""

import numpy as np
def get_asymmetry(mask):
    """
    Calculates assymmetry score for a skin lesion mask and returns a score
     ranging from 0 (perfect symmetry) to 1 (highly assymmetric).
    
    Parameters -> mask  (numpy.ndarray)
    Returns -> Asymmetry score (float) or np.nan if mask invalid
    """
    if mask is None or mask.sum() == 0:
        return np.nan
    mask = (mask > 0)
    
    try:
        y_indices, x_indices = np.where(mask)
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        h, w = mask.shape
        y_min = max(0, center_y - 100)
        y_max = min(h, center_y + 100)
        x_min = max(0, center_x - 100)
        x_max = min(w, center_x + 100)
    
        cropped = mask[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return np.nan
        
        flipped = np.fliplr(cropped)
        min_width = min(cropped.shape[1], flipped.shape[1])
        cropped = cropped[:, :min_width]
        flipped = flipped[:, :min_width]
        
        diff = np.abs(cropped.astype(float) - flipped.astype(float))
        asymmetry = diff.sum() / max(cropped.sum(), 1)
        
        return asymmetry
        
    except Exception:
        return np.nan