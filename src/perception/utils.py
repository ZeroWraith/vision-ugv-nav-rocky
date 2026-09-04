import cv2
import numpy as np

# Professional muted palette (BGR)
CLASS_COLORS = {
    0: (70, 130, 70),    # muted teal-green — traversable
    1: (40, 40, 210),    # muted red — obstacle
    2: (170, 130, 50),   # slate blue — sky
}

# Brighter edge colour for obstacle contours
_EDGE_COLOR = (80, 80, 255)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert single-channel class mask to BGR image."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in CLASS_COLORS.items():
        vis[mask == cls] = col
    return vis


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend colourised mask onto original frame."""
    colored = colorize_mask(mask)
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


def overlay_mask_with_edges(frame: np.ndarray, mask: np.ndarray,
                            alpha: float = 0.35) -> np.ndarray:
    """Blend mask with edge contours around obstacles."""
    colored = colorize_mask(mask)
    blended = cv2.addWeighted(frame, 1.0, colored, alpha, 0)

    # Draw obstacle contours
    obs_binary = (mask == 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(obs_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, _EDGE_COLOR, 1, cv2.LINE_AA)

    return blended
