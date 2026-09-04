import streamlit as st
import numpy as np
import cv2
import time
from pathlib import Path

# ---------- Paths (absolute, no CWD dependency) ----------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "rugd" / "scene_03"
VIDEO_PATH = _PROJECT_ROOT / "demo.mp4"

st.set_page_config(page_title="UGV Vision Nav Demo", layout="wide")

# ---------- Dark theme ----------
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }
    h1 { color: #00d4ff !important; font-size: 1.6rem !important; }
    h2, h3 { color: #00d4ff !important; }
    .stButton>button {
        background-color: #00d4ff; color: #000;
        border-radius: 6px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #00b8e0; }
    .stSlider [data-baseweb=slider] { accent-color: #00d4ff; }
    div[data-testid="stMetric"] {
        background-color: #1a1a2e; border-radius: 8px; padding: 8px;
    }
    div[data-testid="stMetric"] label { color: #888; }
    div[data-testid="stMetric"] div { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

st.title("Vision-Based Autonomous UGV Navigation")
st.caption("RUGD Dataset | DeepLabV3 | PyTorch CUDA | Real-time costmap + A* planning")

# ---------- Determine frame count ----------
if (DATA_DIR / "masks.npy").exists():
    _n = np.load(DATA_DIR / "masks.npy", mmap_mode="r").shape[0]
else:
    _n = 30

TOTAL_FRAMES = _n

# ---------- Session state ----------
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "speed" not in st.session_state:
    st.session_state.speed = 5.0


# ---------- Sidebar ----------
st.sidebar.header("Playback Controls")


def _toggle_play():
    st.session_state.playing = not st.session_state.playing


st.sidebar.button(
    "Pause" if st.session_state.playing else "Play",
    on_click=_toggle_play,
    use_container_width=True,
)

st.session_state.speed = st.sidebar.slider(
    "Speed (FPS)", 1.0, 15.0, st.session_state.speed, 0.5
)


def _on_slider():
    st.session_state.frame_idx = st.session_state.slider_key


st.sidebar.slider(
    "Frame", 0, max(TOTAL_FRAMES - 1, 0),
    value=st.session_state.frame_idx,
    key="slider_key",
    on_change=_on_slider,
)

col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    if st.button("Prev") and st.session_state.frame_idx > 0:
        st.session_state.frame_idx -= 1
with col_next:
    if st.button("Next") and st.session_state.frame_idx < TOTAL_FRAMES - 1:
        st.session_state.frame_idx += 1


# ---------- Load artifacts ----------
@st.cache_data
def load_artifacts():
    masks = np.load(DATA_DIR / "masks.npy")
    poses = np.loadtxt(DATA_DIR / "poses.txt")
    costmap = np.load(DATA_DIR / "costmap.npy")
    origin = np.load(DATA_DIR / "origin.npy")
    waypoints = (np.load(DATA_DIR / "waypoints.npy")
                 if (DATA_DIR / "waypoints.npy").exists() else None)
    cmd_vel = (np.load(DATA_DIR / "cmd_vel.npy")
               if (DATA_DIR / "cmd_vel.npy").exists() else None)
    return masks, poses, costmap, origin, waypoints, cmd_vel


masks, poses, costmap, origin, waypoints, cmd_vel = load_artifacts()
valid = ~np.isnan(poses).any(axis=1)


def render_costmap_frame(idx):
    """Render the costmap with trajectory and A* path."""
    cm_vis = _costmap_colormap_vis(costmap)

    # Trajectory
    traj = poses[valid][:idx + 1]
    for j, p in enumerate(traj):
        gx = int((p[0] - origin[0]) / 0.05)
        gy = int((p[1] - origin[1]) / 0.05)
        if 0 <= gx < cm_vis.shape[1] and 0 <= gy < cm_vis.shape[0]:
            alpha = 0.3 + 0.7 * j / max(len(traj), 1)
            col = tuple(int(c * alpha) for c in (255, 200, 0))
            cv2.circle(cm_vis, (gx, gy), 2, col, -1)

    # Current pose
    if valid[idx]:
        p = poses[idx]
        gx = int((p[0] - origin[0]) / 0.05)
        gy = int((p[1] - origin[1]) / 0.05)
        if 0 <= gx < cm_vis.shape[1] and 0 <= gy < cm_vis.shape[0]:
            cv2.circle(cm_vis, (gx, gy), 5, (0, 255, 0), -1)
            cv2.circle(cm_vis, (gx, gy), 5, (255, 255, 255), 1)

    # A* path
    if waypoints is not None and len(waypoints) > 1:
        pts = []
        for wp in waypoints:
            gx = int((wp[0] - origin[0]) / 0.05)
            gy = int((wp[1] - origin[1]) / 0.05)
            if 0 <= gx < cm_vis.shape[1] and 0 <= gy < cm_vis.shape[0]:
                pts.append([gx, gy])
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(cm_vis, [pts_arr], False, (0, 255, 255), 2, cv2.LINE_AA)

    st.image(cv2.cvtColor(cm_vis, cv2.COLOR_BGR2RGB), use_container_width=True)


def _costmap_colormap_vis(costmap):
    """Gradient costmap visualization."""
    norm = np.clip(costmap.astype(np.float32) / 254.0, 0, 1)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    free = norm < 0.3
    rgb[free, 1] = (norm[free] / 0.3 * 100 + 60).astype(np.uint8)
    rgb[free, 2] = (norm[free] / 0.3 * 40 + 20).astype(np.uint8)

    mid = (norm >= 0.3) & (norm < 0.7)
    t = (norm[mid] - 0.3) / 0.4
    rgb[mid, 1] = (200 - t * 80).astype(np.uint8)
    rgb[mid, 2] = (t * 180).astype(np.uint8)

    high = norm >= 0.7
    rgb[high, 2] = 220
    rgb[high, 1] = (60 - norm[high] * 60).astype(np.uint8)

    unknown = costmap == 127
    rgb[unknown] = (40, 40, 40)
    return rgb


# ---------- Animated fragment ----------
run_every = 1.0 / st.session_state.speed if st.session_state.playing else None


@st.fragment(run_every=run_every)
def synchronized_views():
    idx = st.session_state.frame_idx

    # Auto-advance
    if st.session_state.playing:
        idx = (idx + 1) % TOTAL_FRAMES
        st.session_state.frame_idx = idx

    # ---- Three-column layout ----
    col_cm, col_vid, col_seg = st.columns([1, 2, 1])

    with col_cm:
        st.markdown("**Costmap**")
        render_costmap_frame(idx)

    with col_vid:
        st.markdown("**Camera + Segmentation**")
        vis_frame = _build_vis_frame(idx)
        st.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    with col_seg:
        st.markdown("**Segmentation Mask**")
        from src.perception.utils import colorize_mask
        mask_vis = colorize_mask(masks[idx])
        st.image(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # ---- Metrics row ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame", f"{idx + 1} / {TOTAL_FRAMES}")
    if valid[idx]:
        m2.metric("Pose", f"({poses[idx][0]:.1f}, {poses[idx][1]:.1f})")
    else:
        m2.metric("Pose", "lost")
    if cmd_vel is not None and idx < len(cmd_vel):
        m3.metric("Velocity", f"{cmd_vel[idx][0]:.2f} m/s")
    else:
        m3.metric("Velocity", "—")
    obs_pct = np.mean(masks[idx] == 1) * 100
    m4.metric("Obstacles", f"{obs_pct:.0f}%")


def _build_vis_frame(idx):
    """Build the annotated frame using draw_frame."""
    from src.mapping.viz import draw_frame

    # Load RGB frames from disk (cached)
    if not hasattr(st, "_frame_cache") or st._frame_cache is None:
        st._frame_cache = {}
    if idx not in st._frame_cache:
        rgb_dir = DATA_DIR / "rgb"
        frame_files = sorted(rgb_dir.glob("*.png"))
        if idx < len(frame_files):
            st._frame_cache[idx] = cv2.imread(str(frame_files[idx]))
        else:
            st._frame_cache[idx] = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = st._frame_cache[idx]
    frame_resized = cv2.resize(frame, (640, 360))

    past = poses[valid][:idx + 1] if idx > 0 else None
    cmd = cmd_vel[idx] if cmd_vel is not None and idx < len(cmd_vel) else None

    # Estimate heading from trajectory
    heading = 0.0
    if past is not None and len(past) >= 2:
        dx = past[-1][0] - past[-2][0]
        dy = past[-1][1] - past[-2][1]
        heading = np.arctan2(dx, dy)

    return draw_frame(
        frame_resized, masks[idx],
        poses[idx] if valid[idx] else None,
        waypoints, costmap, origin,
        cmd=cmd, resolution=0.05,
        frame_idx=idx, total_frames=TOTAL_FRAMES,
        inference_ms=0.0,
        past_poses=past, heading=heading,
    )


synchronized_views()

# ---------- Video player ----------
if VIDEO_PATH.exists():
    st.subheader("Demo Video")
    st.video(str(VIDEO_PATH), format="video/mp4")

# ---------- Sidebar info ----------
st.sidebar.markdown("---")
st.sidebar.info(
    f"GPU: NVIDIA T4 | Frames: {TOTAL_FRAMES} | "
    f"Resolution: 640x360 | Costmap: {costmap.shape[0]}x{costmap.shape[1]}"
)
