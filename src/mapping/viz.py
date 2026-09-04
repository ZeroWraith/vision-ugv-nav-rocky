import cv2
import numpy as np
from src.perception.utils import overlay_mask_with_edges

# ---------------------------------------------------------------------------
# Colour constants (BGR)
# ---------------------------------------------------------------------------
_TRAV_COLOR = (70, 130, 70)
_OBS_COLOR = (40, 40, 210)
_SKY_COLOR = (170, 130, 50)
_TRAIL_COLOR = (255, 200, 0)    # cyan
_PATH_COLOR = (0, 255, 255)     # yellow
_GRID_COLOR = (60, 60, 60)
_BORDER_CLEAR = (0, 180, 0)
_BORDER_WARN = (0, 200, 255)
_BORDER_BLOCKED = (0, 0, 220)
_BADGE_BG = (0, 0, 0)
_TEXT_WHITE = (255, 255, 255)
_TEXT_SHADOW = (0, 0, 0)

# Mini-map & legend sizes
_MINI_SIZE = 200
_MINI_PAD = 12
_LEGEND_H = 22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _put_text_shadow(img, text, org, font_scale, color, thickness=1,
                     shadow_off=2):
    """Draw text with a drop-shadow for readability."""
    x, y = org
    cv2.putText(img, text, (x + shadow_off, y + shadow_off),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TEXT_SHADOW,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                thickness, cv2.LINE_AA)


def _costmap_colormap(costmap):
    """Custom gradient: dark-green (free) → yellow → orange → red (lethal)."""
    norm = np.clip(costmap.astype(np.float32) / 254.0, 0, 1)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Free (0) → dark green
    free = norm < 0.3
    rgb[free, 1] = (norm[free] / 0.3 * 100 + 60).astype(np.uint8)
    rgb[free, 2] = (norm[free] / 0.3 * 40 + 20).astype(np.uint8)

    # Mid → yellow/orange
    mid = (norm >= 0.3) & (norm < 0.7)
    t = (norm[mid] - 0.3) / 0.4
    rgb[mid, 1] = (200 - t * 80).astype(np.uint8)
    rgb[mid, 2] = (t * 180).astype(np.uint8)

    # High → red
    high = norm >= 0.7
    rgb[high, 2] = 220
    rgb[high, 1] = (60 - norm[high] * 60).astype(np.uint8)

    # Unknown cells (127 in original) → dark gray
    unknown = costmap == 127
    rgb[unknown] = (40, 40, 40)

    return rgb


# ---------------------------------------------------------------------------
# Main compositor
# ---------------------------------------------------------------------------
def draw_frame(frame, mask, pose, waypoints, costmap, origin, cmd=None,
               resolution=0.05, frame_idx=0, total_frames=0,
               inference_ms=0.0, past_poses=None, heading=0.0):
    """Composite visualisation for a single frame.

    Returns BGR image same size as frame (640x360).
    """
    H, W = frame.shape[:2]
    vis = frame.copy()

    # ------------------------------------------------------------------
    # Layer 1-2: Segmentation overlay with edge contours
    # ------------------------------------------------------------------
    vis = overlay_mask_with_edges(vis, mask, alpha=0.35)

    # ------------------------------------------------------------------
    # Layer 3: Trajectory trail (project past poses onto frame)
    # ------------------------------------------------------------------
    if past_poses is not None and len(past_poses) > 1:
        pts = []
        for pp in past_poses[-20:]:
            if np.isnan(pp).any():
                continue
            # Approximate world-to-screen: robot starts at frame centre
            sx = int(W / 2 + pp[0] * 15)  # scale factor ~15 px/m
            sy = int(H - 20 - pp[1] * 15)
            if 0 <= sx < W and 0 <= sy < H:
                pts.append((sx, sy))
        if len(pts) > 1:
            for j in range(1, len(pts)):
                alpha_t = 0.3 + 0.7 * j / len(pts)
                c = tuple(int(v * alpha_t) for v in _TRAIL_COLOR)
                cv2.line(vis, pts[j - 1], pts[j], c, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(vis, pt, 3, _TRAIL_COLOR, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Layer 4: Faint grid overlay
    # ------------------------------------------------------------------
    grid_step = 80
    for gx in range(0, W, grid_step):
        cv2.line(vis, (gx, 0), (gx, H), _GRID_COLOR, 1)
    for gy in range(0, H, grid_step):
        cv2.line(vis, (0, gy), (W, gy), _GRID_COLOR, 1)

    # ------------------------------------------------------------------
    # Layer 5: Mini-map (top-left)
    # ------------------------------------------------------------------
    mini_x0, mini_y0 = _MINI_PAD, _MINI_PAD + 28  # below info bar
    mini_x1 = mini_x0 + _MINI_SIZE
    mini_y1 = mini_y0 + _MINI_SIZE

    if mini_y1 < H and mini_x1 < W:
        mini_bg = _costmap_colormap(cv2.resize(costmap, (_MINI_SIZE, _MINI_SIZE),
                                                interpolation=cv2.INTER_NEAREST))
        # Draw grid on mini-map
        for g in range(0, _MINI_SIZE, 20):
            cv2.line(mini_bg, (g, 0), (g, _MINI_SIZE), (50, 50, 50), 1)
            cv2.line(mini_bg, (0, g), (_MINI_SIZE, g), (50, 50, 50), 1)

        # Draw A* path on mini-map
        if waypoints is not None and len(waypoints) > 1:
            grid_n = costmap.shape[0]
            path_pts = []
            for wp in waypoints:
                gx = int((wp[0] - origin[0]) / resolution / grid_n * _MINI_SIZE)
                gy = int((wp[1] - origin[1]) / resolution / grid_n * _MINI_SIZE)
                gx = max(0, min(_MINI_SIZE - 1, gx))
                gy = max(0, min(_MINI_SIZE - 1, gy))
                path_pts.append([gx, gy])
            path_arr = np.array(path_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mini_bg, [path_arr], False, _PATH_COLOR, 2, cv2.LINE_AA)

        # Vehicle dot + heading arrow on mini-map
        if pose is not None and not np.isnan(pose).any():
            grid_n = costmap.shape[0]
            vx = int((pose[0] - origin[0]) / resolution / grid_n * _MINI_SIZE)
            vy = int((pose[1] - origin[1]) / resolution / grid_n * _MINI_SIZE)
            vx = max(4, min(_MINI_SIZE - 4, vx))
            vy = max(4, min(_MINI_SIZE - 4, vy))
            cv2.circle(mini_bg, (vx, vy), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(mini_bg, (vx, vy), 5, (255, 255, 255), 1, cv2.LINE_AA)
            # Heading arrow
            arr_len = 14
            ax = int(vx + arr_len * np.sin(heading))
            ay = int(vy - arr_len * np.cos(heading))
            cv2.line(mini_bg, (vx, vy), (ax, ay), (0, 255, 255), 2, cv2.LINE_AA)

        # Semi-transparent border
        cv2.rectangle(mini_bg, (0, 0), (_MINI_SIZE - 1, _MINI_SIZE - 1),
                       (100, 100, 100), 2)

        # Blend onto vis
        roi = vis[mini_y0:mini_y1, mini_x0:mini_x1]
        vis[mini_y0:mini_y1, mini_x0:mini_x1] = cv2.addWeighted(
            roi, 0.3, mini_bg, 0.7, 0)

    # ------------------------------------------------------------------
    # Layer 6: Frame info bar (top)
    # ------------------------------------------------------------------
    bar_h = 26
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

    frame_txt = f"Frame {frame_idx + 1}/{total_frames}"
    fps_txt = f"{inference_ms:.1f}ms/frame" if inference_ms > 0 else ""
    _put_text_shadow(vis, frame_txt, (10, 18), 0.45, _TEXT_WHITE, 1)
    if fps_txt:
        _put_text_shadow(vis, fps_txt, (W - 160, 18), 0.45, _TEXT_WHITE, 1)

    # ------------------------------------------------------------------
    # Layer 7: Status badge (top-right)
    # ------------------------------------------------------------------
    badge_w, badge_h = 140, 22
    badge_x0 = W - badge_w - 10
    badge_y0 = bar_h + 6

    # Determine status from obstacle proximity in mask
    obs_pct = np.mean(mask == 1) * 100 if mask.size > 0 else 0
    if obs_pct > 40:
        status_text = "BLOCKED"
        badge_col = _BORDER_BLOCKED
    elif obs_pct > 10:
        status_text = "OBSTACLES"
        badge_col = _BORDER_WARN
    else:
        status_text = "PATH CLEAR"
        badge_col = _BORDER_CLEAR

    badge_overlay = vis.copy()
    cv2.rectangle(badge_overlay, (badge_x0, badge_y0),
                  (badge_x0 + badge_w, badge_y0 + badge_h), badge_col, -1)
    vis = cv2.addWeighted(badge_overlay, 0.8, vis, 0.2, 0)
    _put_text_shadow(vis, status_text,
                     (badge_x0 + 8, badge_y0 + 16), 0.4, _TEXT_WHITE, 1)

    # ------------------------------------------------------------------
    # Layer 8: Velocity HUD (bottom-left)
    # ------------------------------------------------------------------
    if cmd is not None:
        v, w = cmd
        hud_y = H - 16
        _put_text_shadow(vis, f"v={v:.2f} m/s", (14, hud_y), 0.45,
                         _TEXT_WHITE, 1)
        _put_text_shadow(vis, f"w={w:.2f} rad/s", (140, hud_y), 0.45,
                         _TEXT_WHITE, 1)

    # ------------------------------------------------------------------
    # Layer 9: Legend (bottom-centre)
    # ------------------------------------------------------------------
    legend_items = [
        ("Traversable", _TRAV_COLOR),
        ("Obstacle", _OBS_COLOR),
        ("Sky", _SKY_COLOR),
    ]
    lx = W // 2 - 130
    ly = H - _LEGEND_H - 4
    for label, col in legend_items:
        cv2.rectangle(vis, (lx, ly), (lx + 10, ly + 10), col, -1)
        cv2.rectangle(vis, (lx, ly), (lx + 10, ly + 10), (180, 180, 180), 1)
        _put_text_shadow(vis, label, (lx + 14, ly + 9), 0.32,
                         (200, 200, 200), 1)
        lx += 90

    # ------------------------------------------------------------------
    # Layer 10: Speed gauge (bottom-right, semi-circular arc)
    # ------------------------------------------------------------------
    gauge_cx, gauge_cy = W - 50, H - 40
    gauge_r = 28
    speed_norm = np.clip(abs(cmd[0]) / 1.5, 0, 1) if cmd is not None else 0
    # Background arc
    cv2.ellipse(vis, (gauge_cx, gauge_cy), (gauge_r, gauge_r),
                0, 180, 360, (60, 60, 60), 3, cv2.LINE_AA)
    # Coloured arc
    arc_end = int(180 + speed_norm * 180)
    if speed_norm < 0.5:
        arc_col = (0, 200, 0)
    elif speed_norm < 0.8:
        arc_col = (0, 200, 255)
    else:
        arc_col = (0, 0, 220)
    cv2.ellipse(vis, (gauge_cx, gauge_cy), (gauge_r, gauge_r),
                0, 180, arc_end, arc_col, 3, cv2.LINE_AA)
    _put_text_shadow(vis, f"{abs(cmd[0]):.1f}",
                     (gauge_cx - 10, gauge_cy - 4), 0.35, _TEXT_WHITE, 1)
    _put_text_shadow(vis, "m/s", (gauge_cx - 10, gauge_cy + 10), 0.28,
                     (160, 160, 160), 1)

    # ------------------------------------------------------------------
    # Layer 11: Heading compass (below status badge)
    # ------------------------------------------------------------------
    compass_cx = badge_x0 + badge_w // 2
    compass_cy = badge_y0 + badge_h + 22
    compass_r = 14
    cv2.circle(vis, (compass_cx, compass_cy), compass_r, (50, 50, 50), -1)
    cv2.circle(vis, (compass_cx, compass_cy), compass_r, (120, 120, 120), 1)
    # Arrow in heading direction
    ax = int(compass_cx + (compass_r - 3) * np.sin(heading))
    ay = int(compass_cy - (compass_r - 3) * np.cos(heading))
    cv2.line(vis, (compass_cx, compass_cy), (ax, ay), (0, 200, 255), 2,
             cv2.LINE_AA)
    # "N" label
    _put_text_shadow(vis, "N", (compass_cx - 3, compass_cy - compass_r - 3),
                     0.28, (180, 180, 180), 1)

    # ------------------------------------------------------------------
    # Layer 12: Coloured border
    # ------------------------------------------------------------------
    if obs_pct > 40:
        bcol = _BORDER_BLOCKED
    elif obs_pct > 10:
        bcol = _BORDER_WARN
    else:
        bcol = _BORDER_CLEAR
    cv2.rectangle(vis, (0, 0), (W - 1, H - 1), bcol, 3)

    return vis
