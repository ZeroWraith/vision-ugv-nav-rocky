import streamlit as st
import numpy as np
import cv2
import os
from pathlib import Path

st.set_page_config(page_title="UGV Vision Nav Demo", layout="wide")

st.title("🛰️ Vision‑Based Autonomous Navigation (RUGD‑rocky)")

# ---------- Sidebar ----------
st.sidebar.header("Controls")
show_video = st.sidebar.checkbox("Show demo video", True)
show_costmap = st.sidebar.checkbox("Show cost‑map explorer", True)
frame_idx = st.sidebar.slider("Frame", 0, 299, 0)

DATA_DIR = Path("../data/rugd/scene_03")
VIDEO_PATH = Path("demo.mp4")

# ---------- Load artifacts ----------
@st.cache_data
def load_artifacts():
    masks = np.load(DATA_DIR / "masks.npy")
    poses = np.loadtxt(DATA_DIR / "poses.txt")
    costmap = np.load(DATA_DIR / "costmap.npy")
    origin = np.load(DATA_DIR / "origin.npy")
    waypoints = np.load(DATA_DIR / "waypoints.npy") if (DATA_DIR / "waypoints.npy").exists() else None
    return masks, poses, costmap, origin, waypoints

masks, poses, costmap, origin, waypoints = load_artifacts()

# ---------- Video ----------
if show_video and VIDEO_PATH.exists():
    st.subheader("🎬 Demo video")
    st.video(str(VIDEO_PATH))

# ---------- Cost‑map explorer ----------
if show_costmap:
    st.subheader("🗺️ Cost‑map & trajectory")
    col1, col2 = st.columns([2,1])
    with col1:
        # render costmap with trajectory up to frame_idx
        cm_vis = cv2.applyColorMap(costmap.astype(np.uint8), cv2.COLORMAP_JET)
        # draw trajectory
        valid = ~np.isnan(poses).any(axis=1)
        traj = poses[valid][:frame_idx+1]
        for p in traj:
            gx = int((p[0] - origin[0]) / 0.05)
            gy = int((p[1] - origin[1]) / 0.05)
            if 0 <= gx < cm_vis.shape[1] and 0 <= gy < cm_vis.shape[0]:
                cv2.circle(cm_vis, (gx, gy), 1, (0,255,0), -1)
        # current pose
        if valid[frame_idx]:
            p = poses[frame_idx]
            gx = int((p[0] - origin[0]) / 0.05)
            gy = int((p[1] - origin[1]) / 0.05)
            cv2.circle(cm_vis, (gx, gy), 4, (255,255,0), -1)
        # waypoints
        if waypoints is not None:
            for wp in waypoints:
                gx = int((wp[0] - origin[0]) / 0.05)
                gy = int((wp[1] - origin[1]) / 0.05)
                if 0 <= gx < cm_vis.shape[1] and 0 <= gy < cm_vis.shape[0]:
                    cv2.circle(cm_vis, (gx, gy), 2, (255,0,255), -1)
        st.image(cv2.cvtColor(cm_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.markdown(f"**Frame:** {frame_idx}")
        st.markdown(f"**Pose (x,y):** {poses[frame_idx][:2] if valid[frame_idx] else 'lost'}")
        if waypoints is not None:
            st.markdown(f"**Waypoints:** {len(waypoints)}")
        # show segmentation mask for current frame
        mask = masks[frame_idx]
        from src.perception.utils import colorize_mask
        mask_vis = colorize_mask(mask)
        st.image(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB), caption="Segmentation", use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Built for hackathon – runs entirely in Colab (GPU).")