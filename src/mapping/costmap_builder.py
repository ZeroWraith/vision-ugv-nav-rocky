import numpy as np
import cv2
from tqdm.auto import tqdm

def _pixel_to_world(u, v, depth, K, cam_height, pitch):
    """Back‑project a pixel (u,v) with assumed ground‑plane depth to world XY."""
    # Normalised camera coordinates
    x_cam = (u - K[0,2]) * depth / K[0,0]
    y_cam = (v - K[1,2]) * depth / K[1,1]
    z_cam = depth

    # Rotate by pitch (camera pitched down)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x_w = x_cam
    y_w = cp * y_cam - sp * z_cam
    # world Z = cp*z_cam + sp*y_cam  (not needed for 2‑D)
    return x_w, y_w

def build_costmap(masks, poses, K, cam_height=1.2, pitch=0.0,
                  resolution=0.05, grid_size_m=80.0, inflate_radius=0.3):
    """
    masks   : (N, H, W) uint8 {0 traversable,1 obstacle,2 sky}
    poses   : (N, 3) world XY (translation only)
    Returns costmap (grid, grid) uint8, origin (world xy of grid[0,0])
    """
    grid_n = int(grid_size_m / resolution)
    costmap = np.full((grid_n, grid_n), 127, dtype=np.uint8)   # unknown
    origin = np.array([-grid_size_m/2.0, -grid_size_m/2.0])    # centre at (0,0)

    # Pre‑compute pixel grid for mask (only traversable class)
    H, W = masks.shape[1], masks.shape[2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    for i in tqdm(range(len(masks)), desc="Building costmap"):
        pose = poses[i]
        if np.isnan(pose).any():
            continue
        x_robot, y_robot = pose[0], pose[1]

        # Assume ground plane at camera height -> depth = cam_height / sin(pitch+...?)
        # For simplicity use constant depth = cam_height / tan(pitch+small)
        # Here we approximate depth = cam_height / np.tan(pitch + 0.1) if pitch>0 else 5.0
        if pitch > 0:
            depth = cam_height / np.tan(pitch)
        else:
            depth = 5.0   # arbitrary forward look distance

        # Convert traversable pixels
        trav = (masks[i] == 0)
        if not trav.any():
            continue
        u_t = uu[trav]
        v_t = vv[trav]

        xs, ys = _pixel_to_world(u_t, v_t, depth, K, cam_height, pitch)

        # Transform to world
        xs_w = xs + x_robot
        ys_w = ys + y_robot

        # Grid indices
        gx = ((xs_w - origin[0]) / resolution).astype(int)
        gy = ((ys_w - origin[1]) / resolution).astype(int)

        valid = (gx >= 0) & (gx < grid_n) & (gy >= 0) & (gy < grid_n)
        gx, gy = gx[valid], gy[valid]
        costmap[gy, gx] = 0   # free

        # Obstacles (class 1) -> mark occupied
        obs = (masks[i] == 1)
        if obs.any():
            u_o, v_o = uu[obs], vv[obs]
            xo, yo = _pixel_to_world(u_o, v_o, depth, K, cam_height, pitch)
            xo_w = xo + x_robot
            yo_w = yo + y_robot
            gox = ((xo_w - origin[0]) / resolution).astype(int)
            goy = ((yo_w - origin[1]) / resolution).astype(int)
            valid_o = (gox >=0)&(gox<grid_n)&(goy>=0)&(goy<grid_n)
            costmap[goy[valid_o], gox[valid_o]] = 255

    # Inflate obstacles
    if inflate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (int(inflate_radius/resolution)*2+1,
                                            int(inflate_radius/resolution)*2+1))
        occ = (costmap == 255).astype(np.uint8)
        occ = cv2.dilate(occ, kernel)
        costmap[occ == 1] = 255

    return costmap, origin