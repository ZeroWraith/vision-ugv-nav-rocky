import cv2
import numpy as np

# colour map for 3 classes: 0 traversable (green), 1 obstacle (red), 2 sky (blue)
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
}

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert single‑channel class mask to BGR image for overlay."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in CLASS_COLORS.items():
        vis[mask == cls] = col
    return vis

def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend colourised mask onto original frame."""
    colored = colorize_mask(mask)
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)