import numpy as np
import cv2
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt


def _pixel_depths(v_coords, W, H, K, cam_height):
    """Per-pixel ground-plane depth using FOV-based model (works for any pitch).

    Returns depth array same shape as v_coords, in metres.
    """
    fy = K[1, 1]
    cy = K[1, 2]
    half_vfov = np.arctan2(H / 2.0, fy)

    # Angle below camera horizontal for each row
    beta = half_vfov * (H - 1 - v_coords) / (H / 2.0)
    beta = np.clip(beta, 0.01, np.pi / 2 - 0.01)

    depth = cam_height / np.tan(beta)
    return np.clip(depth, 0.3, 80.0)


def _pixel_to_world(u, v, depth, K):
    """Back-project pixel (u,v) with known depth to camera-frame XY."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    return x_cam, depth  # (lateral, forward)


def build_costmap(masks, poses, K, cam_height=1.2, pitch=0.0,
                  resolution=0.05, grid_size_m=80.0, inflate_radius=0.3):
    """
    Build a gradient occupancy costmap from semantic masks + robot poses.

    masks   : (N, H, W) uint8  {0 traversable, 1 obstacle, 2 sky}
    poses   : (N, 3) world XY (translation only)
    K       : (3, 3) camera intrinsics
    Returns costmap (grid_n, grid_n) uint8, origin (2,) world-XY of grid[0,0].
    """
    grid_n = int(grid_size_m / resolution)
    costmap = np.full((grid_n, grid_n), 127, dtype=np.uint8)   # unknown
    origin = np.array([-grid_size_m / 2.0, -grid_size_m / 2.0])

    H, W = masks.shape[1], masks.shape[2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    # Pre-compute per-row depth
    depth_map = _pixel_depths(np.arange(H), W, H, K, cam_height)

    n = min(len(masks), len(poses))
    for i in tqdm(range(n), desc="Building costmap"):
        pose = poses[i]
        if np.isnan(pose).any():
            continue
        x_robot, y_robot = pose[0], pose[1]

        # ---- Traversable ----
        trav = (masks[i] == 0)
        if trav.any():
            u_t = uu[trav]
            v_t = vv[trav]
            d_t = depth_map[v_t]

            x_cam, y_fwd = _pixel_to_world(u_t, v_t, d_t, K)

            xs_w = x_cam + x_robot
            ys_w = y_fwd + y_robot

            gx = ((xs_w - origin[0]) / resolution).astype(int)
            gy = ((ys_w - origin[1]) / resolution).astype(int)

            valid = (gx >= 0) & (gx < grid_n) & (gy >= 0) & (gy < grid_n)
            costmap[gy[valid], gx[valid]] = 0

        # ---- Obstacles ----
        obs = (masks[i] == 1)
        if obs.any():
            u_o = uu[obs]
            v_o = vv[obs]
            d_o = depth_map[v_o]

            x_cam_o, y_fwd_o = _pixel_to_world(u_o, v_o, d_o, K)

            xo_w = x_cam_o + x_robot
            yo_w = y_fwd_o + y_robot

            gox = ((xo_w - origin[0]) / resolution).astype(int)
            goy = ((yo_w - origin[1]) / resolution).astype(int)

            valid_o = (gox >= 0) & (gox < grid_n) & (goy >= 0) & (goy < grid_n)
            costmap[goy[valid_o], gox[valid_o]] = 254

    # ---- Exponential gradient inflation ----
    obs_mask = (costmap >= 254)
    dist = distance_transform_edt(~obs_mask) * resolution  # metres to nearest obstacle
    gradient = (254.0 * np.exp(-3.0 * np.clip(dist, 0.0, 3.0))).astype(np.uint8)

    # Blend: gradient fills unknown cells with proximity cost, obstacle centres stay lethal
    result = np.maximum(gradient, costmap)
    result[obs_mask] = 254

    # Clip unknown (127) to stay as unknown — gradient should only fill explored area
    explored = (costmap != 127)
    final = np.where(explored, result, 127).astype(np.uint8)

    return final, origin
