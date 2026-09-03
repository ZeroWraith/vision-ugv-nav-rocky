"""
Thin wrapper around pyslam.ORBSLAM3 for monocular use.
Assumes the vocabulary file is installed by pyslam at the default location.
"""
from pyslam import ORBSLAM3
import numpy as np

class OrbSlam3Mono:
    def __init__(self, config_path: str):
        vocab_path = "/usr/local/share/orb_slam3/ORBvoc.txt"
        self.slam = ORBSLAM3(vocab_path, config_path)

    def track(self, gray_frame: np.ndarray, timestamp: float):
        """
        Process a single monocular frame.
        Returns 4x4 pose matrix (world->camera) or None if tracking lost.
        """
        return self.slam.track_monocular(gray_frame, timestamp)

    def shutdown(self):
        self.slam.shutdown()