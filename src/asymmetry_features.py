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
    if mask.dtype != bool:
        mask = mask > 0.5
    try:
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return np.nan
        min_y, max_y = y_indices.min(), y_indices.max()
        min_x, max_x = x_indices.min(), x_indices.max()
        center_y = int((min_y + max_y) / 2)
        center_x = int((min_x + max_x) / 2)
        lesion_height = max_y - min_y
        lesion_width = max_x - min_x
        crop_half = max(lesion_height, lesion_width) // 2 + 20
        h, w = mask.shape
        y_min = max(0, center_y - crop_half)
        y_max = min(h, center_y + crop_half)
        x_min = max(0, center_x - crop_half)
        x_max = min(w, center_x + crop_half)
        cropped = mask[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return np.nan
        flipped = np.fliplr(cropped)
        min_width = min(cropped.shape[1], flipped.shape[1])
        cropped = cropped[:, :min_width]
        flipped = flipped[:, :min_width]
        diff = np.abs(cropped.astype(float) - flipped.astype(float))
        asymmetry = diff.sum() / max(cropped.sum(), 1)

        asymmetry = min(asymmetry, 1.0)      
        return round(float(asymmetry), 4)
    except Exception:
        return np.nan
