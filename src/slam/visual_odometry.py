import cv2
import numpy as np


class SimpleVisualOdometry:
    """Frame-to-frame monocular visual odometry using OpenCV Essential Matrix.

    Produces (N, 3) [x, y, yaw] pose array compatible with the pipeline.
    Scale is assumed constant (user-supplied SCALE metres per unit translation).

    Parameters
    ----------
    K : (3, 3) float64
        Camera intrinsic matrix.
    scale : float
        Metres per frame-step. Handles monocular scale ambiguity.
    """

    def __init__(self, K, scale=0.5):
        self.K = np.array(K, dtype=np.float64)
        self.scale = scale
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_gray = None
        self.prev_pose = np.zeros(3, dtype=np.float64)  # [x, y, yaw]
        self.poses = [self.prev_pose.copy()]

    def process_frame(self, frame_bgr):
        """Process a single BGR frame and return (x, y, yaw).

        Returns the current pose. If tracking fails (not enough features),
        returns the last known pose without moving.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return self.prev_pose.copy()

        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            self.prev_gray = gray
            return self.prev_pose.copy()

        # Match features
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 10:
            self.prev_gray = gray
            return self.prev_pose.copy()

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Essential matrix + recover pose
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or mask is None:
            self.prev_gray = gray
            return self.prev_pose.copy()

        inliers = mask.ravel().sum()
        if inliers < 8:
            self.prev_gray = gray
            return self.prev_pose.copy()

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Extract yaw from rotation matrix (z-axis rotation for ground plane)
        yaw_delta = np.arctan2(R[1, 0], R[0, 0])

        # Scale translation to metric
        dx = self.scale * t[0, 0]
        dy = self.scale * t[2, 0]  # camera Z = forward direction

        # Rotate delta by current yaw to get world-frame displacement
        cos_y = np.cos(self.prev_pose[2])
        sin_y = np.sin(self.prev_pose[2])
        wx = cos_y * dx - sin_y * dy
        wy = sin_y * dx + cos_y * dy

        new_pose = np.array([
            self.prev_pose[0] + wx,
            self.prev_pose[1] + wy,
            self.prev_pose[2] + yaw_delta,
        ])

        self.prev_gray = gray
        self.prev_pose = new_pose
        self.poses.append(new_pose.copy())

        return new_pose

    def get_poses(self):
        """Return all accumulated poses as (N, 3) array [x, y, yaw]."""
        return np.array(self.poses, dtype=np.float64)
