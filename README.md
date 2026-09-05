# Vision-Based Autonomous Navigation for UGV

An end-to-end vision-based autonomous navigation pipeline for an Unmanned Ground Vehicle (UGV) operating in unstructured outdoor environments without GPS. The system processes real camera frames from the **RUGD dataset** across three distinct terrain scenarios — **Creek** (rock bed), **Village** (buildings, roads), and **Trail** (forest path) — through five pipeline stages: **Perception → Localization → Mapping → Planning → Control**, producing navigable paths, velocity commands, and polished demo videos.

Runs entirely in **Google Colab (free T4 GPU)** or locally on any machine with Python 3.10+.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerowraith/vision-ugv-nav-rocky/blob/main/notebooks/01_full_pipeline.ipynb)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
  - [Stage 1: Dataset Acquisition](#stage-1-dataset-acquisition)
  - [Stage 2: Perception (Semantic Segmentation)](#stage-2-perception-semantic-segmentation)
  - [Stage 3: Localization (Visual Odometry)](#stage-3-localization-visual-odometry)
  - [Stage 4: Mapping (Costmap Construction)](#stage-4-mapping-costmap-construction)
  - [Stage 5: Global Planning (A\*)](#stage-5-global-planning-a)
  - [Stage 6: Local Control (Pure Pursuit)](#stage-6-local-control-pure-pursuit)
  - [Stage 7: Visualization (Demo Video)](#stage-7-visualization-demo-video)
  - [Stage 8: Interactive UI (Streamlit)](#stage-8-interactive-ui-streamlit)
- [Multi-Terrain Scenarios](#multi-terrain-scenarios)
- [Repository Layout](#repository-layout)
- [Source Code Reference](#source-code-reference)
- [Data Artifacts](#data-artifacts)
- [Camera Intrinsics](#camera-intrinsics)
- [GPU Support](#gpu-support)
- [Configuration Parameters](#configuration-parameters)
- [Dependencies](#dependencies)
- [License and Citation](#license-and-citation)

---

## Quick Start

### Google Colab (Recommended)

1. Click the Colab badge above.
2. **Runtime → Change runtime type → GPU (T4)**.
3. **Run all cells** (~10 min including ~2 min download + ~6 min segmentation across 3 terrains).
4. Three individual demo videos + one montage video are produced.
5. A public **ngrok URL** appears at the end — open it for the interactive Streamlit UI with scene selector.

### Local

```bash
git clone https://github.com/ZeroWraith/vision-ugv-nav-rocky.git
cd vision-ugv-nav-rocky
pip install -r requirements.txt
jupyter notebook notebooks/01_full_pipeline.ipynb
```

Run all cells. When prompted, optionally launch the Streamlit UI.

---

## Pipeline Architecture

The notebook executes 7 sequential stages. Each stage reads from and writes to per-scene directories under `data/rugd/`.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE DATA FLOW                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                        │
│  │  RUGD ZIP    │───▶│  RGB Frames  │───▶│  DeepLabV3   │                        │
│  │  (5.3 GB)    │    │  + meta.json │    │  PyTorch CUDA│                        │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                        │
│                                                  │                               │
│                                           masks.npy (N,H,W)                      │
│                                                  │                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  ORB Feature │───▶│  Essential   │───▶│  Pose        │                   │
│  │  Detection   │    │  Matrix +    │    │  Integration │                   │
│  │              │    │  recoverPose │    │  (x,y,yaw)   │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                          │
│                                           poses.txt (N,3)                  │
│                                                  ▼                               │
│                                          ┌──────────────┐                        │
│                                          │  Costmap     │                        │
│                                          │  Builder     │                        │
│                                          │  (gradient)  │                        │
│                                          └──────┬───────┘                        │
│                                                  │                               │
│                                    costmap.npy + origin.npy                      │
│                                                  │                               │
│                                                  ▼                               │
│                                          ┌──────────────┐                        │
│                                          │  A* Global   │                        │
│                                          │  (cost-aware)│                        │
│                                          └──────┬───────┘                        │
│                                                  │                               │
│                                          waypoints.npy                           │
│                                                  │                               │
│                                                  ▼                               │
│                                          ┌──────────────┐                        │
│                                          │ Pure Pursuit │                        │
│                                          │ Local Ctrl   │                        │
│                                          └──────┬───────┘                        │
│                                                  │                               │
│                                          cmd_vel.npy                             │
│                                                  │                               │
│                                      ┌───────────┴───────────┐                   │
│                                      ▼                       ▼                   │
│                              ┌──────────────┐       ┌──────────────┐             │
│                              │  Per-scene   │       │  Streamlit   │             │
│                              │  demo.mp4    │       │  UI          │             │
│                              └──────┬───────┘       └──────────────┘             │
│                                     │                                            │
│                                     ▼                                            │
│                             ┌──────────────┐                                     │
│                             │  Montage     │                                     │
│                             │  demo.mp4    │                                     │
│                             └──────────────┘                                     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Dataset Acquisition

**What:** Downloads the RUGD (Robot Unstructured Ground Driving) dataset and extracts three terrain sequences for multi-scenario demo.

**Why:** Different terrains demonstrate the pipeline's versatility — rock bed (Creek), man-made structures (Village), and dense forest (Trail).

**How it works:**

1. Checks if all three scene directories already contain frames.
2. If any are missing, downloads `http://rugd.vision/data/RUGD_frames-with-annotations.zip` (~5.3 GB) using `wget` (with `urllib` fallback).
3. Extracts frames for three scenes in a single pass through the zip:
   - `creek/` → `data/rugd/scene_03/rgb/` (~836 frames)
   - `village/` → `data/rugd/village/rgb/` (~117 frames)
   - `trail-7/` → `data/rugd/trail_7/rgb/` (~290 frames)
4. Renames frames to sequential format: `frame_0000.png`, `frame_0001.png`, etc.
5. Writes `meta.json` with RUGD camera calibration (identical for all scenes).
6. Cleans up the downloaded zip file.

**Key files produced:**
- `data/rugd/{scene_dir}/rgb/frame_XXXX.png` — raw camera frames (1376×1110 native, used at 640×480)
- `data/rugd/{scene_dir}/meta.json` — camera intrinsics and configuration

**Code:** Notebook Cell 2

---

### Stage 2: Perception (Semantic Segmentation)

**What:** Classifies every pixel in each camera frame into one of three semantic classes: **traversable ground** (0), **obstacle** (1), or **sky** (2).

**Why:** The robot needs to distinguish walkable terrain from obstacles and sky to navigate safely.

**How it works:**

1. **Model loading (Cell 3):** Loads `torchvision.models.segmentation.deeplabv3_resnet50` with pretrained COCO/VOC weights directly onto the CUDA GPU. The model is loaded once and shared across all three terrain scenarios.

2. **Inference (Cell 3, `segment_frame()`):** For each frame:
   - Resizes to 640×360 (model input size).
   - Normalizes pixel values with ImageNet statistics: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
   - Transposes to NCHW format: `(1, 3, 360, 640)`.
   - Runs PyTorch inference with `torch.no_grad()`.
   - Applies `argmax` across the 21 VOC class channels.
   - Remaps 21 VOC classes to 3 navigation classes.
   - Returns a `(360, 640)` uint8 mask.

3. **Caching:** Masks are saved per-scene as `masks.npy` and reused on subsequent runs.

**Class mapping (VOC → Navigation):**

| Nav Value | Nav Class | Color (overlay) | VOC Classes Included |
|-----------|-----------|-----------------|---------------------|
| 0 | Traversable | Green `(70, 130, 70)` | background(0), road(7), grass(10), sidewalk(11), terrain(12), fence(13), dirt(14) |
| 1 | Obstacle | Red `(40, 40, 210)` | person(15), car(6), boulder, wall, etc. (all others) |
| 2 | Sky | Blue `(170, 130, 50)` | sky(17) |

**Code:** `src/perception/fastscnn_onnx.py` (`DeepLabV3Seg` class), `src/perception/utils.py` (visualization), Notebook Cell 3

---

### Stage 3: Localization (Visual Odometry)

**What:** Estimates the robot's pose (position + yaw) for each frame using frame-to-frame visual odometry with ORB features and the Essential Matrix.

**Why:** To build a consistent map, the system needs to know where the robot was when each frame was captured. Visual odometry uses the camera's own motion between frames to estimate trajectory — no GPS required.

**How it works:**

1. **Feature detection (`cv2.ORB_create(nfeatures=3000)`):** Detects ORB keypoints in consecutive grayscale frames.

2. **Feature matching (`cv2.BFMatcher`):** Matches ORB descriptors between frames using Hamming distance, filtered by Lowe's ratio test (0.7).

3. **Essential Matrix estimation (`cv2.findEssentialMat`):** Computes the Essential Matrix from matched point correspondences using RANSAC (threshold=1.0, confidence=0.999).

4. **Pose recovery (`cv2.recoverPose`):** Decomposes the Essential Matrix into rotation `R` and unit translation `t`, returning the number of inlier correspondences.

5. **Pose integration:**
   - **Yaw delta:** Extracted from rotation matrix: `arctan2(R[1,0], R[0,0])`
   - **Translation:** Scaled by a fixed constant (`SCALE` metres per frame-step) to handle monocular scale ambiguity.
   - **World-frame displacement:** Rotated by current yaw: `wx = cos(yaw)*dx - sin(yaw)*dy`, `wy = sin(yaw)*dx + cos(yaw)*dy`
   - **Accumulated:** `new_pose = prev_pose + [wx, wy, yaw_delta]`

6. **Per-scene scale factors:**

   | Scene | Scale (m/frame) | Rationale |
   |-------|----------------|-----------|
   | Creek | 0.5 | Moderate forward speed on rocky terrain |
   | Village | 0.6 | Faster on paved roads |
   | Trail | 0.4 | Slower on rough forest path |

7. **Fallback:** If VO produces < 2m total movement (e.g., featureless scene), falls back to simulated curved trajectory from `src/sim/trajectory.py`.

**Code:** `src/slam/visual_odometry.py` (`SimpleVisualOdometry` class), Notebook Cell 3

---

### Stage 4: Mapping (Costmap Construction)

**What:** Projects segmented pixels from each camera frame into world coordinates to build a 2D gradient occupancy costmap.

**Why:** The robot needs a bird's-eye-view map with continuous cost values to plan paths that prefer low-cost terrain and maintain safety margins around obstacles.

**How it works:**

1. **Grid setup:** Creates a 1600×1600 cell grid (80m × 80m at 0.05m/cell resolution), initialized to `127` (unknown).

2. **Per-pixel depth estimation (`_pixel_depths()`):** Instead of a constant depth, each pixel row gets its own depth based on its vertical position:
   ```python
   half_vfov = arctan2(H/2, fy)
   beta = half_vfov * (H - 1 - v) / (H/2)    # angle below horizon
   depth = cam_height / tan(beta)               # ground-plane distance
   depth = clip(depth, 0.3, 80.0)
   ```
   Pixels at the bottom of the image (close) get small depth; pixels near the horizon get large depth.

3. **For each frame:**
   - Extracts traversable pixels (class 0) and obstacle pixels (class 1).
   - **Back-projects** each pixel to world coordinates using per-pixel depth and camera intrinsics.
   - Transforms world coordinates by the robot's pose.
   - Maps to grid indices and marks cells as free (`0`) or occupied (`254`).

4. **Exponential gradient inflation:** After processing all frames, computes Euclidean distance to nearest obstacle using `scipy.ndimage.distance_transform_edt`, then applies exponential decay:
   ```python
   dist = distance_transform_edt(~obs_mask) * resolution  # metres
   gradient = 254 * exp(-3.0 * clip(dist, 0, 3.0))
   ```
   This produces a smooth cost gradient: `254` (lethal) → `200` (danger) → `50` (near) → `0` (free).

**Code:** `src/mapping/costmap_builder.py`, Notebook Cell 3

---

### Stage 5: Global Planning (A*)

**What:** Finds a cost-aware collision-free path from the robot's current position to a goal position using A* pathfinding on the gradient costmap.

**Why:** Given the occupancy grid with continuous costs, the robot needs a global route that avoids obstacles and prefers low-cost terrain.

**How it works:**

1. **Start position:** First valid pose from trajectory (world XY).
2. **Goal position:** 15 meters ahead along the direction from start to final pose.
3. **A* algorithm:**
   - Uses 8-connected grid (cardinal + diagonal moves).
   - Heuristic: Euclidean distance (`np.hypot`).
   - Base move costs: 1.0 for cardinal, 1.414 (√2) for diagonal.
   - **Cost-aware edge weights:** `edge_weight = base_cost × (1.0 + cell_cost / 254.0)` — paths through high-cost terrain (near obstacles) are penalized.
   - Skips cells where `costmap >= 250` (lethal obstacle).
   - Unknown cells (`127`) and low-cost gradient cells are passable.
4. **Fallbacks:**
   - If start or goal is on a lethal cell → returns straight line `[start, goal]`.
   - If no path exists → returns straight line `[start, goal]`.
5. Returns list of world-frame `(x, y)` waypoints.

**Code:** `src/planning/astar_planner.py`, Notebook Cell 3

---

### Stage 6: Local Control (Pure Pursuit)

**What:** Converts the global waypoint path into velocity commands `(v, w)` for a differential-drive robot using the Pure Pursuit algorithm.

**Why:** The global planner gives a route, but the robot needs real-time velocity commands to follow it smoothly.

**How it works:**

1. **Target selection:** Finds the first waypoint at distance ≥ `lookahead` (1.5m) ahead of the robot. If none found, uses the last waypoint.
2. **Frame transformation:** Transforms the target from world frame to vehicle-local frame using the current yaw.
3. **Behind-target handling:** If `local_x ≤ 0` (target behind robot), sets `v = 0` and rotates in place.
4. **Curvature calculation:**
   ```
   L = hypot(local_x, local_y)
   curvature = 2 × local_y / (L²)
   v = max_v (1.5 m/s)
   w = clip(v × curvature, -max_w, max_w)   # max_w = 1.0 rad/s
   ```

**Code:** `src/planning/pure_pursuit.py`, Notebook Cell 3

---

### Stage 7: Visualization (Demo Video)

**What:** Renders a professional 12-layer composite visualization for each frame, writing per-scene `demo.mp4` videos and a side-by-side montage.

**Why:** Provides a visual record of the pipeline's output with all computed data overlays, suitable for hackathon presentation.

**How it works:**

For each frame, `draw_frame()` composites:

| # | Layer | Position | Content | Style |
|---|-------|----------|---------|-------|
| 1 | Segmentation overlay | Full frame | Colorized mask with edge contours | Muted palette (green/red/blue), 35% alpha, obstacle contours |
| 2 | Trajectory trail | Main view | Last 20 vehicle positions | Cyan dots + thin polyline |
| 3 | Grid overlay | Main view | Faint reference grid | White, 1px every 80px |
| 4 | Mini-map | Top-left, 200×200 | Gradient costmap + grid + A* path + vehicle dot with heading arrow | Custom terrain colormap, semi-transparent |
| 5 | Frame info bar | Top, full width | `Frame 15/30 | 12.3ms/frame` | Dark bar, white text |
| 6 | Status badge | Top-right | `PATH CLEAR` / `OBSTACLES` / `BLOCKED` | Color-coded pill badge |
| 7 | Velocity HUD | Bottom-left | `v=0.50 m/s  w=0.12 rad/s` | White text with shadow |
| 8 | Legend | Bottom-center | 3 colored squares + labels | Small, unobtrusive |
| 9 | Speed gauge | Bottom-right | Semi-circular arc showing speed | White arc with green/yellow/red fill |
| 10 | Heading compass | Below status badge | Arrow showing yaw direction | White circle with yellow arrow |
| 11 | Border | Full frame | 3px colored border | Green = clear, Yellow = obstacles nearby, Red = blocked |

**Title card:** Each video begins with a 2-second title card showing the terrain name and tech stack, with a fade-in animation.

**Montage video:** All per-scene videos are stitched side-by-side using ffmpeg `xstack` filter, producing a single comparison video.

**Video encoding:** Raw BGR frames are piped directly to ffmpeg via stdin (`-f rawvideo -pix_fmt bgr24`), encoded to H.264 (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`). No intermediate files or OpenCV VideoWriter.

**Code:** `src/mapping/viz.py` (`draw_frame()`), `src/perception/utils.py` (`overlay_mask_with_edges()`), Notebook Cells 3, 5

---

### Stage 8: Interactive UI (Streamlit)

**What:** Launches a web-based interactive dashboard for exploring the pipeline results across all three terrains.

**Why:** Allows real-time inspection of individual frames, costmaps, trajectories, and segmentation masks with animated playback.

**How it works:**

- **Colab mode:** Starts Streamlit in a background thread, creates an ngrok tunnel to port 8501, prints the public URL.
- **Local mode:** Prints launch instructions, optionally starts Streamlit interactively.

**UI features:**

- **Scene selector** — Dropdown in sidebar to switch between Creek, Village, and Trail.
- **Animated playback** — Play/Pause button with adjustable speed (1-15 FPS) using `@st.fragment(run_every=...)`.
- **Synchronized views** — Three-column layout:
  - Left: Gradient costmap with trajectory, A* path, and vehicle dot.
  - Center: Camera frame with segmentation overlay + edge contours.
  - Right: Colorized segmentation mask.
- **Performance metrics** — Frame number, pose coordinates, velocity, obstacle percentage.
- **Dark theme** — Professional dark background with cyan accents.
- **Video player** — Watch per-scene `demo.mp4` directly in the browser.
- **Manual controls** — Frame slider, Prev/Next buttons.

**Code:** `src/ui/streamlit_app.py`, Notebook Cell 6

---

## Multi-Terrain Scenarios

The pipeline runs on three distinct RUGD terrain sequences:

| Scene | Directory | Frames | Terrain Type | Trajectory Style |
|-------|-----------|--------|-------------|-----------------|
| **Creek** | `scene_03` | 836 | Rock bed, water, boulders | Wide turns (amplitude=10m) |
| **Village** | `village` | ~117 | Buildings, paved roads, fences | Tight turns (amplitude=6m) |
| **Trail** | `trail_7` | ~290 | Forest path, gravel, trees | Frequent turns (amplitude=12m) |

Each scene produces its own set of artifacts (`masks.npy`, `poses.txt`, `costmap.npy`, `waypoints.npy`, `cmd_vel.npy`, `demo.mp4`) and is independently viewable in the Streamlit UI.

---

## Repository Layout

```
vision-ugv-nav-rocky/
├── data/                           # RUGD data (git-ignored)
│   ├── README.md                   # Dataset attribution
│   └── rugd/
│       ├── scene_03/               # Creek sequence
│       │   ├── rgb/                # Extracted creek frames
│       │   │   ├── frame_0000.png
│       │   │   └── ...
│       │   ├── meta.json           # Camera intrinsics
│       │   ├── masks.npy           # Segmentation masks (N,H,W)
│       │   ├── poses.txt           # Robot poses (N,3)
│       │   ├── costmap.npy         # Gradient costmap (G,G)
│       │   ├── origin.npy          # Grid origin (2,)
│       │   ├── waypoints.npy       # A* path (M,2)
│       │   ├── cmd_vel.npy         # Velocity commands (N,2)
│       │   └── demo.mp4            # Per-scene demo video
│       ├── village/                # Village sequence (same structure)
│       └── trail_7/                # Trail sequence (same structure)
│
├── notebooks/
│   └── 01_full_pipeline.ipynb      # Main pipeline notebook (7 cells)
│
├── src/
│   ├── __init__.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── fastscnn_onnx.py        # DeepLabV3 PyTorch segmentation
│   │   └── utils.py                # Mask visualization + edge contours
│   ├── sim/
│   │   ├── __init__.py
│   │   └── trajectory.py           # Curved trajectory generator
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── costmap_builder.py      # Gradient costmap builder
│   │   └── viz.py                  # 12-layer frame compositor
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── astar_planner.py        # Cost-aware A* pathfinding
│   │   └── pure_pursuit.py         # Pure pursuit controller
│   ├── slam/
│   │   ├── __init__.py
│   │   ├── visual_odometry.py       # ORB + Essential Matrix VO
│   │   └── orb_slam3_wrapper.py     # ORB-SLAM3 wrapper (optional)
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py        # Interactive web dashboard
│   └── pyslam/                     # pyslam submodule (optional)
│
├── scripts/
│   └── download_rugd.sh            # Standalone download script
│
├── demo_montage.mp4                # Side-by-side all terrains
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE                         # MIT
└── README.md                       # This file
```

---

## Source Code Reference

### `src/perception/fastscnn_onnx.py`

PyTorch DeepLabV3-ResNet50 segmentation wrapper for CUDA GPU inference.

| Symbol | Type | Description |
|--------|------|-------------|
| `DeepLabV3Seg` | Class | PyTorch GPU inference wrapper |
| `DeepLabV3Seg.__init__(input_size=(640,360))` | Method | Loads model with pretrained weights, moves to CUDA |
| `DeepLabV3Seg.infer(bgr_frame) → np.ndarray` | Method | Runs inference, returns `(H,W)` uint8 mask (0=traversable, 1=obstacle, 2=sky) |

**Class remapping:**
```python
_TRAVERSABLE = {0, 7, 10, 11, 12, 13, 14}  # VOC classes → nav class 0
_SKY = {17}                                   # VOC class 17 → nav class 2
# Everything else → nav class 1 (obstacle)
```

### `src/perception/utils.py`

Visualization utilities for segmentation masks.

| Symbol | Type | Description |
|--------|------|-------------|
| `CLASS_COLORS` | Dict | `{0: (70,130,70), 1: (40,40,210), 2: (170,130,50)}` BGR |
| `colorize_mask(mask) → np.ndarray` | Function | Converts `(H,W)` class mask to `(H,W,3)` BGR image |
| `overlay_mask(frame, mask, alpha=0.35) → np.ndarray` | Function | Alpha-blends colorized mask onto frame |
| `overlay_mask_with_edges(frame, mask, alpha=0.35) → np.ndarray` | Function | Alpha-blends mask + draws obstacle edge contours |

### `src/slam/visual_odometry.py`

Frame-to-frame monocular visual odometry using OpenCV Essential Matrix.

| Symbol | Type | Description |
|--------|------|-------------|
| `SimpleVisualOdometry` | Class | ORB + Essential Matrix VO wrapper |
| `SimpleVisualOdometry.__init__(K, scale=0.5)` | Method | Initializes with camera intrinsics and fixed scale |
| `SimpleVisualOdometry.process_frame(frame_bgr) → np.ndarray` | Method | Processes BGR frame, returns `(x, y, yaw)` pose |
| `SimpleVisualOdometry.get_poses() → np.ndarray` | Method | Returns all accumulated poses as `(N, 3)` array |

**Algorithm:**
1. ORB feature detection (3000 features)
2. BFMatcher with Hamming distance + Lowe's ratio test (0.7)
3. `cv2.findEssentialMat` (RANSAC, threshold=1.0)
4. `cv2.recoverPose` → R, t
5. Yaw from `arctan2(R[1,0], R[0,0])`
6. Scaled translation integrated in world frame

### `src/sim/trajectory.py`

Simulated curved trajectory (used as fallback when VO fails).

| Symbol | Type | Description |
|--------|------|-------------|
| `simulate_trajectory(n_frames, speed, turn_freq, amplitude, start_angle) → np.ndarray` | Function | Returns `(N,3)` poses `[x, y, yaw]` with sinusoidal weaving |
| `SCENE_TRAJECTORIES` | Dict | Per-scene trajectory parameters |

### `src/mapping/costmap_builder.py`

Builds gradient 2D occupancy grid from segmentation masks and robot poses.

| Symbol | Type | Description |
|--------|------|-------------|
| `_pixel_depths(v_coords, W, H, K, cam_height) → np.ndarray` | Function | Per-pixel ground-plane depth using FOV-based model |
| `_pixel_to_world(u, v, depth, K) → tuple` | Function | Back-projects pixel to camera-frame XY |
| `build_costmap(masks, poses, K, cam_height, pitch, resolution, grid_size_m, inflate_radius) → tuple` | Function | Returns `(costmap, origin)` |

**Parameters:**
- `masks`: `(N, H, W)` uint8 — segmentation masks
- `poses`: `(N, 3)` float — robot world positions `[x, y, yaw]`
- `K`: `(3, 3)` float — camera intrinsic matrix
- `resolution`: meters per grid cell (default 0.05)
- `grid_size_m`: physical grid size in meters (default 80.0)
- `inflate_radius`: obstacle inflation radius in meters (default 0.3)

**Returns:**
- `costmap`: `(grid_n, grid_n)` uint8 — gradient values `0`=free, `127`=unknown, `1-254`=cost gradient, `254`=lethal
- `origin`: `(2,)` float — world XY of `costmap[0,0]`

### `src/mapping/viz.py`

12-layer frame visualization compositor.

| Symbol | Type | Description |
|--------|------|-------------|
| `draw_frame(frame, mask, pose, waypoints, costmap, origin, cmd, resolution, frame_idx, total_frames, inference_ms, past_poses, heading) → np.ndarray` | Function | Creates annotated frame with all visual overlays |

### `src/planning/astar_planner.py`

Cost-aware A* pathfinding on 2D occupancy grid.

| Symbol | Type | Description |
|--------|------|-------------|
| `_heuristic(a, b) → float` | Function | Euclidean distance between two grid points |
| `astar(costmap, origin, start_xy, goal_xy, resolution) → list` | Function | Returns list of world `(x,y)` waypoints |

**Behavior:**
- 8-connected grid with diagonal costs = √2
- Lethal cells (`>= 250`) are impassable
- Cost-aware edge weights: `edge = base_cost × (1.0 + cell_cost / 254.0)`
- Unknown cells (`127`) and low-cost gradient cells are passable
- Falls back to straight line if start/goal blocked or no path found

### `src/planning/pure_pursuit.py`

Pure Pursuit local path-tracking controller.

| Symbol | Type | Description |
|--------|------|-------------|
| `pure_pursuit_step(pose, waypoints, lookahead, max_v, max_w) → tuple` | Function | Returns `(v, w)` velocity commands |

**Parameters:**
- `pose`: `(x, y, yaw)` — current robot pose (yaw in radians)
- `waypoints`: list of `(x, y)` — global path from A*
- `lookahead`: target distance in meters (default 1.5)
- `max_v`: max linear velocity in m/s (default 1.5)
- `max_w`: max angular velocity in rad/s (default 1.0)

### `src/ui/streamlit_app.py`

Interactive web dashboard with animated playback and multi-scene support.

| Symbol | Type | Description |
|--------|------|-------------|
| `SCENES` | Dict | Scene label → directory mapping |
| `load_artifacts(_scene_dir)` | Cached Function | Loads masks, poses, costmap, origin, waypoints, cmd_vel per scene |
| `synchronized_views()` | Fragment | Animated view with `@st.fragment(run_every=...)` |
| `_build_vis_frame(idx)` | Function | Builds annotated frame using `draw_frame()` |
| `render_costmap_frame(idx)` | Function | Renders gradient costmap with trajectory and A* path |

**UI sections:**
- Scene selector dropdown (Creek / Village / Trail)
- Play/Pause with speed control (1-15 FPS)
- Synchronized costmap + camera + segmentation columns
- Frame info, pose, velocity, obstacle metrics
- Per-scene video player
- Dark theme

### `src/slam/orb_slam3_wrapper.py`

Thin wrapper around pyslam's ORB-SLAM3 for monocular visual SLAM (optional, not used in default pipeline).

| Symbol | Type | Description |
|--------|------|-------------|
| `OrbSlam3Mono` | Class | ORB-SLAM3 wrapper |
| `OrbSlam3Mono.__init__(config_path)` | Method | Loads ORB vocabulary and config |
| `OrbSlam3Mono.track(gray_frame, timestamp) → np.ndarray or None` | Method | Processes frame, returns 4×4 pose matrix |
| `OrbSlam3Mono.shutdown()` | Method | Cleans up SLAM system |

---

## Data Artifacts

All artifacts are saved per-scene to `data/rugd/{scene_dir}/`:

| File | Format | Shape / Size | Contents |
|------|--------|-------------|----------|
| `rgb/frame_XXXX.png` | PNG | 1376×1110×3 | Raw camera frames from RUGD |
| `meta.json` | JSON | — | Camera intrinsics and config |
| `masks.npy` | NumPy | `(N, 360, 640)` uint8 | Per-frame segmentation masks |
| `poses.txt` | Text | `(N, 3)` float | Robot poses `[x, y, yaw]` (curved trajectory) |
| `costmap.npy` | NumPy | `(1600, 1600)` uint8 | Gradient occupancy grid (0-254) |
| `origin.npy` | NumPy | `(2,)` float | World XY of grid[0,0] |
| `waypoints.npy` | NumPy | `(M, 2)` float | A* path waypoints |
| `cmd_vel.npy` | NumPy | `(N, 2)` float | Velocity commands `[v, w]` |
| `demo.mp4` | MP4 H.264 | 640×360 @ 10fps | Per-scene demo video with title card |

Additionally, `demo_montage.mp4` (in repo root) contains all three scenes stitched side-by-side.

---

## Camera Intrinsics

The RUGD dataset uses a Prosilica GT2750C camera with an 8mm lens mounted on a Clearpath Husky robot. The calibration values used (identical for all scenes):

```
Intrinsic Matrix K:
┌                     ┐
│  381.362   0   320.5 │    fx = fy = 381.362  (focal length in pixels)
│    0     381.362 240.5│    cx = 320.5        (principal point x)
│    0       0       1 │    cy = 240.5        (principal point y)
└                     ┘

Distortion: [0, 0, 0, 0, 0]  (no distortion correction applied)
Camera height: 1.2 meters above ground
Pitch angle: 0.0 degrees (horizontal)
Image resolution: 1376×1110 (native), used at 640×480
```

---

## GPU Support

The pipeline uses GPU acceleration at two stages:

| Stage | Library | GPU Usage |
|-------|---------|-----------|
| **Segmentation** | PyTorch | `model.cuda()` — DeepLabV3 inference on CUDA |
| **Costmap** | SciPy | `distance_transform_edt` on CPU (fast for 1600×1600) |

**Detection logic:**
```python
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU REQUIRED but not detected.')
GPU_NAME = torch.cuda.get_device_name(0)
```

GPU is **required** — the pipeline raises an error if no GPU is detected. This ensures consistent performance for the demo.

---

## Configuration Parameters

### Pipeline Parameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `speed` | `0.5` m/frame | Cell 3 | Base trajectory speed |
| `turn_freq` | `0.015-0.03` | Cell 3 | Sinusoidal turn frequency |
| `amplitude` | `6-12` m | Cell 3 | Lateral weaving amplitude |
| `resolution` | `0.05` m/cell | Cell 3 | Costmap grid resolution |
| `grid_size_m` | `80.0` m | Cell 3 | Physical size of costmap |
| `inflate_radius` | `0.3` m | Cell 3 | Obstacle safety margin |
| `lookahead` | `1.5` m | Cell 3 | Pure pursuit lookahead distance |
| `max_v` | `1.5` m/s | Cell 3 | Maximum linear velocity |
| `max_w` | `1.0` rad/s | Cell 3 | Maximum angular velocity |
| `video_fps` | `10.0` | Cell 3 | Output video frame rate |
| `TARGET_W × TARGET_H` | `640 × 360` | Cell 3 | Output video resolution |

### Camera Parameters (meta.json)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `width` | 640 | Image width (pixels) |
| `height` | 480 | Image height (pixels) |
| `fps` | 10 | Frame rate |
| `K[0,0]` / `K[1,1]` | 381.362 | Focal length (pixels) |
| `K[0,2]` | 320.5 | Principal point cx (pixels) |
| `K[1,2]` | 240.5 | Principal point cy (pixels) |
| `camera_height` | 1.2 | Camera height above ground (m) |
| `pitch_deg` | 0.0 | Camera pitch angle (degrees) |

---

## Dependencies

### Python Packages (`requirements.txt`)

```
torch                           # PyTorch (DeepLabV3 inference, GPU)
torchvision                     # Vision models (DeepLabV3-ResNet50)
torchaudio                      # Audio processing (PyTorch dependency)
opencv-python-headless          # Image I/O and processing
numpy                           # Array operations
scipy                           # Distance transform for costmap inflation
matplotlib                      # Plotting (optional)
tqdm                            # Progress bars
streamlit                       # Web UI framework
pyngrok                         # ngrok tunnel for Colab
gdown                           # Google Drive downloader
```

### System Packages (Colab)

```
cmake, build-essential          # C++ compilation (optional pyslam)
libopencv-dev                   # OpenCV headers
wget, unzip                     # Dataset download/extraction
ffmpeg                          # Video encoding (H.264 via libx264)
```

---

## License and Citation

### Code
MIT License — see `LICENSE`.

### RUGD Dataset
CC-BY-4.0. If you use this data, please cite:

```bibtex
@inproceedings{RUGD2019IROS,
  author    = {Wigness, Maggie and Eum, Sungmin and Rogers, John G and Han, David and Kwon, Heesung},
  title     = {A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments},
  booktitle = {International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2019}
}
```

### DeepLabV3
```bibtex
@inproceedings{chen2017rethinking,
  title     = {Rethinking Atrous Convolution for Semantic Image Segmentation},
  author    = {Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L},
  booktitle = {arXiv preprint arXiv:1706.05587},
  year      = {2017}
}
```

### ORB-SLAM3
```bibtex
@article{orbcolon_slam3,
  title     = {ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multimap SLAM},
  author    = {Campos, Carlos and Elvira, Richard and Rodr{\'i}guez, Juan J. G. and Montiel, Jos{\'e} M. M. and Tard{\'o}s, Juan D.},
  journal   = {IEEE Transactions on Robotics},
  year      = {2021}
}
```
