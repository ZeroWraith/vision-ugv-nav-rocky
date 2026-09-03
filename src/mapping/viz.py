import cv2
import numpy as np

def draw_frame(frame, mask, pose, waypoints, costmap, origin, cmd=None,
               resolution=0.05):
    """
    Composite visualisation for a single frame.
    Returns BGR image same size as frame.
    """
    vis = frame.copy()

    # 1) Segmentation overlay (green traversable, red obstacle)
    from src.perception.utils import overlay_mask
    vis = overlay_mask(vis, mask, alpha=0.35)

    # 2) Draw cost‑map mini‑map in corner (120x120)
    mini = cv2.resize(costmap, (120,120), interpolation=cv2.INTER_NEAREST)
    mini_color = cv2.applyColorMap(mini, cv2.COLORMAP_JET)
    vis[10:130, 10:130] = cv2.addWeighted(vis[10:130,10:130], 0.5, mini_color, 0.5, 0)

    # 3) Vehicle pose on mini‑map
    if pose is not None and not np.isnan(pose).any():
        gx = int((pose[0] - origin[0]) / resolution)
        gy = int((pose[1] - origin[1]) / resolution)
        if 0 <= gx < 120 and 0 <= gy < 120:
            cv2.circle(mini_color, (gx, gy), 3, (0,255,0), -1)
            vis[10:130,10:130] = mini_color

    # 4) Global path on main image (project first few waypoints)
    if waypoints is not None and len(waypoints) > 0:
        for wp in waypoints[:30]:
            # simple projection assuming flat ground and same camera model as mapping
            # skip – just draw on mini‑map for clarity
            pass

    # 5) HUD text
    if cmd is not None:
        v, w = cmd
        cv2.putText(vis, f'v={v:.2f} m/s  w={w:.2f} rad/s',
                    (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

    return vis