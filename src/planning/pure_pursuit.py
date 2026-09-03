import numpy as np

def pure_pursuit_step(pose, waypoints, lookahead=1.5, max_v=1.5, max_w=1.0):
    """
    pose      : (x, y, yaw)   yaw in radians
    waypoints : list of (x,y) in world frame (global path)
    Returns (v, w) linear & angular velocity.
    """
    if len(waypoints) == 0:
        return 0.0, 0.0

    x, y, yaw = pose
    # Find target point at lookahead distance along path
    target = None
    for wp in waypoints:
        dx = wp[0] - x
        dy = wp[1] - y
        dist = np.hypot(dx, dy)
        if dist >= lookahead:
            target = wp
            break
    if target is None:
        target = waypoints[-1]

    # Transform target to vehicle frame
    tx, ty = target[0] - x, target[1] - y
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x = cos_yaw * tx + sin_yaw * ty
    local_y = -sin_yaw * tx + cos_yaw * ty

    if local_x <= 0:
        # target behind, just rotate towards it
        angle = np.arctan2(local_y, local_x)
        v = 0.0
        w = np.clip(angle, -max_w, max_w)
        return v, w

    # Curvature = 2 * y / L^2
    L = np.hypot(local_x, local_y)
    curvature = 2.0 * local_y / (L * L)
    v = max_v
    w = np.clip(v * curvature, -max_w, max_w)
    return v, w