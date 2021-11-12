import numpy as np

def grayscale_variation(image: np.ndarray, eta: int = 10, g_min: int = 155, g_max: int = 255, near_depth_thresh: int = 0, far_depth_thresh: int = None) -> np.ndarray:
    """Quantizes depth levels into discrete grayscale image levels.

    Args:
        image (np.ndarray): Input image of shape (H, W).
        eta (int, optional): Number of gray levels. Defaults to 10.
        g_min (int, optional): Lowest gray level. Defaults to 155.
        g_max (int, optional): Highest gray level. Defaults to 255.
        near_depth_thresh (int, optional): Minimum considered depth. Defaults to 0.
        far_depth_thresh (int, optional): Maximum considered depth. Defaults to None.
    Returns:
        np.ndarray: Depth quantized image.
    """
    
    mask = image > near_depth_thresh
    if far_depth_thresh != None:
        mask = np.logical_and(mask, image < far_depth_thresh)
    
    d_max = max(1, image.max())
    if np.any(mask):
        d_min = image[mask].min()
    else:
        d_min = 0
    
    g_stride = int((g_max - g_min) / eta)
    return np.where(mask, g_min + np.round(eta * (image - d_min) / (d_max - d_min)) * g_stride, 0).astype(np.uint8)