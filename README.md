# Vision-Based Autonomous Navigation for UGV (Outdoor, GPS-Denied)

An end-to-end vision-based autonomous navigation pipeline for an Unmanned Ground Vehicle (UGV) operating in unstructured outdoor environments without GPS. The system processes real camera frames from the **RUGD creek sequence** (rocky terrain) through five pipeline stages: **Perception → Localization → Mapping → Planning → Control**, producing a navigable path and velocity commands.

Runs entirely in **Google Colab (free T4 GPU)** or locally on any machine with Python 3.10+.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerowraith/vision-ugv-nav-rocky/blob/main/notebooks/01_full_pipeline.ipynb)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
  - [Stage 1: Dataset Acquisition](#stage-1-dataset-acquisition)
  - [Stage 2: Perception (Semantic Segmentation)](#stage-2-perception-semantic-segmentation)
  - [Stage 3: Visual Localization (SLAM)](#stage-3-visual-localization-slam)
  - [Stage 4: Mapping (Costmap Construction)](#stage-4-mapping-costmap-construction)
  - [Stage 5: Global Planning (A\*)](#stage-5-global-planning-a)
  - [Stage 6: Local Control (Pure Pursuit)](#stage-6-local-control-pure-pursuit)
  - [Stage 7: Visualization (Demo Video)](#stage-7-visualization-demo-video)
  - [Stage 8: Interactive UI (Streamlit)](#stage-8-interactive-ui-streamlit)
- [Repository Layout](#repository-layout)
- [Source Code Reference](#source-code-reference)
- [Data Artifacts](#data-artifacts)
- [Camera Intrinsics](#camera-intrinsics)
- [GPU Support](#gpu-support)
- [Configuration Parameters](#configuration-parameters)
- [Dependencies](#dependencies)
- [Docker](#docker)
- [License and Citation](#license-and-citation)

---

## Quick Start

### Google Colab (Recommended)

1. Click the Colab badge above.
2. **Runtime → Change runtime type → GPU (T4)**.
3. **Run all cells** (~4 min including ~2 min download).
4. A public **ngrok URL** appears at the end — open it for the interactive Streamlit UI.
5. `demo.mp4` is also saved and downloadable.

### Local

```bash
git clone https://github.com/ZeroWraith/vision-ugv-nav-rocky.git
cd vision-ugv-nav-rocky
pip install -r requirements.txt
jupyter notebook notebooks/01_full_pipeline.ipynb
```

Run all cells. When prompted, optionally launch the Streamlit UI.

### Docker

```bash
docker build -t ugv-nav .
docker run --gpus all -p 8888:8888 ugv-nav
# Open http://localhost:8888 and run the notebook
```

---

## Pipeline Architecture

The notebook executes 8 sequential stages. Each stage reads from and writes to the `data/rugd/scene_03/` directory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE DATA FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  RUGD ZIP    │───▶│  RGB Frames  │───▶│  FastSCNN    │                   │
│  │  (5.3 GB)    │    │  + meta.json │    │  ONNX GPU    │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                          │
│                                           masks.npy (N,H,W)                 │
│                                                  │                          │
│  ┌──────────────┐    ┌──────────────┐           │                          │
│  │  pyslam /    │───▶│  poses.txt   │───────────┤                          │
│  │  simulated   │    │  (N,3)       │           │                          │
│  └──────────────┘    └──────────────┘           │                          │
│                                                  ▼                          │
│                                          ┌──────────────┐                   │
│                                          │  Costmap     │                   │
│                                          │  Builder     │                   │
│                                          └──────┬───────┘                   │
│                                                  │                          │
│                                    costmap.npy + origin.npy                 │
│                                                  │                          │
│                                                  ▼                          │
│                                          ┌──────────────┐                   │
│                                          │  A* Global   │                   │
│                                          │  Planner     │                   │
│                                          └──────┬───────┘                   │
│                                                  │                          │
│                                          waypoints.npy                      │
│                                                  │                          │
│                                                  ▼                          │
│                                          ┌──────────────┐                   │
│                                          │ Pure Pursuit │                   │
│                                          │ Local Ctrl   │                   │
│                                          └──────┬───────┘                   │
│                                                  │                          │
│                                          cmd_vel.npy                        │
│                                                  │                          │
│                                      ┌───────────┴───────────┐              │
│                                      ▼                       ▼              │
│                              ┌──────────────┐       ┌──────────────┐        │
│                              │  demo.mp4    │       │  Streamlit   │        │
│                              │  Video       │       │  UI          │        │
│                              └──────────────┘       └──────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Dataset Acquisition

**What:** Downloads the RUGD (Robot Unstructured Ground Driving) dataset and extracts the creek sequence — the only sequence with significant rocky/rock-bed terrain.

**Why:** The pipeline needs real-world outdoor camera frames with known camera calibration to produce meaningful navigation results.

**How it works:**

1. The notebook checks if `data/rugd/scene_03/rgb/` already contains PNG frames.
2. If empty, downloads `http://rugd.vision/data/RUGD_frames-with-annotations.zip` (~5.3 GB) using `wget` (with `urllib` fallback).
3. Extracts **only** the `creek/*.png` frames from the zip (not all 18 scenes) using Python's `zipfile` module.
4. Renames frames to sequential format: `frame_0000.png`, `frame_0001.png`, etc.
5. Creates `meta.json` with the RUGD camera calibration parameters.
6. Cleans up the downloaded zip file.

**Key files produced:**
- `data/rugd/scene_03/rgb/frame_XXXX.png` — raw camera frames (1376×1110 native, used at 640×480)
- `data/rugd/scene_03/meta.json` — camera intrinsics and configuration

**Code:** Notebook Cell 2, `scripts/download_rugd.sh`

---

### Stage 2: Perception (Semantic Segmentation)

**What:** Classifies every pixel in each camera frame into one of three semantic classes: **traversable ground** (0), **obstacle** (1), or **sky** (2).

**Why:** The robot needs to distinguish walkable terrain from obstacles and sky to navigate safely. This is the core perception task for off-road navigation.

**How it works:**

1. **Model export (Cell 4):** If `models/fastscnn_rugd.onnx` doesn't exist, exports a torchvision DeepLabV3-ResNet50 model to ONNX format:
   - Wraps the model in a `SegWrapper` that extracts the `'out'` key from the output dict.
   - Exports with `torch.onnx.export()` using opset 17, dynamic batch axis, input shape `(1, 3, 360, 640)`.
   - Uses GPU if available (`torch.cuda.is_available()`).

2. **Inference (Cell 5):** The `FastSCNN` class loads the ONNX model:
   - Probes `ort.get_available_providers()` for CUDA support.
   - If `CUDAExecutionProvider` is available, uses GPU; otherwise falls back to `CPUExecutionProvider` with a warning.
   - For each frame:
     - Resizes to 640×360 (model input size).
     - Normalizes pixel values to `[0, 1]` float32.
     - Transposes to NCHW format: `(1, 3, 360, 640)`.
     - Runs ONNX Runtime inference.
     - Applies `argmax` across the channel axis to get per-pixel class IDs.
     - Returns a `(360, 640)` uint8 mask.
   - All masks are stacked into `(N, 360, 640)` and saved to `masks.npy`.

**Class mapping:**
| Value | Class | Color (overlay) | Meaning |
|-------|-------|-----------------|---------|
| 0 | Traversable | Green `(0, 255, 0)` | Safe to drive (grass, dirt, rock) |
| 1 | Obstacle | Red `(0, 0, 255)` | Cannot traverse (boulders, walls) |
| 2 | Sky | Blue `(255, 0, 0)` | Overhead (not ground) |

**Code:** `src/perception/fastscnn_onnx.py` (FastSCNN class), `src/perception/utils.py` (visualization), Notebook Cells 4-5

---

### Stage 3: Visual Localization (SLAM)

**What:** Estimates the robot's 3D pose (position + orientation) for each frame using visual odometry.

**Why:** To build a consistent map, the system needs to know where the robot was when each frame was captured. Without GPS, this must be done视觉ly.

**How it works:**

1. Attempts to import `pyslam.visual_odometry.VisualOdometry` (ORB-based visual odometry from the pyslam library).
2. If pyslam is available, processes each grayscale frame through `slam.track(frame)` to get camera poses.
3. If pyslam is not available (common — it requires specific C++ build deps), **falls back to simulated trajectory:**
   - Assumes constant forward motion along the Z-axis: `z = i * 0.5` meters per frame.
   - This produces a straight-line trajectory that still demonstrates the mapping and planning stages.
4. Poses are saved as `(N, 3)` text file (x, y, z translations).

**Note:** The pyslam ORB-SLAM3 wrapper (`src/slam/orb_slam3_wrapper.py`) exists for production use but requires the full ORB-SLAM3 library to be compiled and installed. The notebook's fallback ensures the pipeline runs in all environments.

**Code:** `src/slam/orb_slam3_wrapper.py`, Notebook Cell 6

---

### Stage 4: Mapping (Costmap Construction)

**What:** Projects segmented pixels from each camera frame into world coordinates to build a 2D occupancy grid (costmap).

**Why:** The robot needs a bird's-eye-view map of its environment to plan collision-free paths. Each pixel classified as "traversable" or "obstacle" is back-projected onto the ground plane.

**How it works:**

1. **Grid setup:** Creates a 1600×1600 cell grid (80m × 80m at 0.05m/cell resolution), initialized to `127` (unknown).

2. **For each frame:**
   - Extracts traversable pixels (class 0) and obstacle pixels (class 1).
   - **Back-projects** each pixel to world coordinates using the camera intrinsics:
     - Converts pixel `(u, v)` to normalized camera coordinates using `K`.
     - Applies pitch rotation to transform from camera frame to world frame.
     - Depth is approximated: `depth = cam_height / tan(pitch)` if pitch > 0, else 5.0m.
   - Transforms world coordinates by the robot's pose (adds robot position).
   - Maps to grid indices and marks cells as free (`0`) or occupied (`255`).

3. **Obstacle inflation:** After processing all frames, dilates occupied cells using morphological dilation with an elliptical structuring element (radius = `inflate_radius / resolution` cells). This creates a safety margin around obstacles.

**The `_pixel_to_world` function:**
```python
def _pixel_to_world(u, v, depth, K, cam_height, pitch):
    x_cam = (u - K[0,2]) * depth / K[0,0]   # horizontal offset
    y_cam = (v - K[1,2]) * depth / K[1,1]   # vertical offset
    z_cam = depth
    cp, sp = np.cos(pitch), np.sin(pitch)
    x_w = x_cam
    y_w = cp * y_cam - sp * z_cam             # pitch rotation
    return x_w, y_w
```

**Code:** `src/mapping/costmap_builder.py`, Notebook Cell 7

---

### Stage 5: Global Planning (A*)

**What:** Finds a collision-free path from the robot's current position to a goal position using A* pathfinding on the costmap.

**Why:** Given the occupancy grid, the robot needs a global route that avoids all obstacles.

**How it works:**

1. **Start position:** First valid pose from SLAM (world XY).
2. **Goal position:** 12 meters ahead along the X-axis from start.
3. **A* algorithm:**
   - Uses 8-connected grid (cardinal + diagonal moves).
   - Heuristic: Euclidean distance (`np.hypot`).
   - Move costs: 1.0 for cardinal, 1.414 (√2) for diagonal.
   - Skips cells where `costmap == 255` (occupied).
   - Unknown cells (`127`) are treated as passable.
4. **Fallbacks:**
   - If start or goal is on an occupied cell → returns straight line `[start, goal]`.
   - If no path exists → returns straight line `[start, goal]`.
5. Returns list of world-frame `(x, y)` waypoints.

**Code:** `src/planning/astar_planner.py`, Notebook Cell 8

---

### Stage 6: Local Control (Pure Pursuit)

**What:** Converts the global waypoint path into velocity commands `(v, w)` for a differential-drive robot using the Pure Pursuit algorithm.

**Why:** The global planner gives a route, but the robot needs real-time velocity commands to follow it smoothly.

**How it works:**

1. **Target selection:** Finds the first waypoint at distance ≥ `lookahead` (1.5m) ahead of the robot. If none found, uses the last waypoint.
2. **Frame transformation:** Transforms the target from world frame to vehicle-local frame using the current yaw:
   ```
   local_x = cos(yaw) * dx + sin(yaw) * dy
   local_y = -sin(yaw) * dx + cos(yaw) * dy
   ```
3. **Behind-target handling:** If `local_x ≤ 0` (target behind robot), sets `v = 0` and rotates in place.
4. **Curvature calculation:**
   ```
   L = hypot(local_x, local_y)
   curvature = 2 * local_y / (L * L)
   v = max_v (1.5 m/s)
   w = clip(v * curvature, -max_w, max_w)   # max_w = 1.0 rad/s
   ```

**Code:** `src/planning/pure_pursuit.py`, Notebook Cell 8

---

### Stage 7: Visualization (Demo Video)

**What:** Renders a composite visualization for each frame and writes it to `demo.mp4`.

**Why:** Provides a visual record of the pipeline's output, combining the original view with all computed data overlays.

**How it works:**

For each frame, `draw_frame()` composites:

1. **Segmentation overlay** — Colorized mask alpha-blended onto the original frame (green=traversable, red=obstacle, blue=sky) at 35% opacity.
2. **Costmap mini-map** — 120×120 pixel JET-colormap version of the costmap placed in the top-left corner, with:
   - Green dot for current robot position.
   - Cyan dot for current pose.
3. **HUD text** — Bottom-left velocity readout: `v=XX.XX m/s  w=XX.XX rad/s`.

**Output:** `demo.mp4` at 640×360, 10 FPS.

**Code:** `src/mapping/viz.py` (draw_frame), `src/perception/utils.py` (overlay_mask, colorize_mask), Notebook Cell 9

---

### Stage 8: Interactive UI (Streamlit)

**What:** Launches a web-based interactive dashboard for exploring the pipeline results.

**Why:** Allows real-time inspection of individual frames, costmaps, trajectories, and segmentation masks.

**How it works:**

- **Colab mode:** Starts Streamlit in a background thread, creates an ngrok tunnel to port 8501, prints the public URL.
- **Local mode:** Prints launch instructions, optionally starts Streamlit interactively.

**UI features:**
- **Video player** — Watch `demo.mp4` directly in the browser.
- **Costmap explorer** — Two-column layout:
  - Left: Costmap with trajectory (green), current pose (cyan), and A* waypoints (magenta).
  - Right: Current frame number, pose coordinates, waypoint count, and colorized segmentation mask.
- **Sidebar controls** — Toggle video/costmap display, scrub through frames with a slider.

**Code:** `src/ui/streamlit_app.py`, Notebook Cell 10

---

## Repository Layout

```
vision-ugv-nav-rocky/
├── data/                           # RUGD data (git-ignored)
│   ├── README.md                   # Dataset attribution
│   └── rugd/scene_03/
│       ├── rgb/                    # Extracted creek frames
│       │   ├── frame_0000.png
│       │   ├── frame_0001.png
│       │   └── ...
│       ├── meta.json               # Camera intrinsics
│       ├── masks.npy               # Segmentation masks (N,H,W)
│       ├── poses.txt               # Robot poses (N,3)
│       ├── costmap.npy             # Occupancy grid (G,G)
│       ├── origin.npy              # Grid origin (2,)
│       ├── waypoints.npy           # A* path (M,2)
│       └── cmd_vel.npy             # Velocity commands (N,2)
│
├── notebooks/
│   └── 01_full_pipeline.ipynb      # Main pipeline notebook
│
├── src/
│   ├── __init__.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── fastscnn_onnx.py        # ONNX segmentation wrapper
│   │   └── utils.py                # Mask visualization utilities
│   ├── slam/
│   │   ├── __init__.py
│   │   └── orb_slam3_wrapper.py    # ORB-SLAM3 monocular wrapper
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── costmap_builder.py      # 2D occupancy grid builder
│   │   └── viz.py                  # Frame visualization compositor
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── astar_planner.py        # A* pathfinding
│   │   └── pure_pursuit.py         # Pure pursuit controller
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py        # Interactive web dashboard
│   └── pyslam/                     # pyslam submodule (optional)
│
├── models/
│   └── fastscnn_rugd.onnx          # Exported segmentation model
│
├── scripts/
│   └── download_rugd.sh            # Standalone download script
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # GPU Docker image
├── .gitignore
├── LICENSE                         # MIT
└── README.md                       # This file
```

---

## Source Code Reference

### `src/perception/fastscnn_onnx.py`

ONNX Runtime wrapper for Fast-SCNN semantic segmentation.

| Symbol | Type | Description |
|--------|------|-------------|
| `FastSCNN` | Class | ONNX inference wrapper |
| `FastSCNN.__init__(onnx_path, input_size=(640,360))` | Method | Loads ONNX model, selects GPU/CPU provider |
| `FastSCNN.infer(bgr_frame) → np.ndarray` | Method | Runs inference, returns `(H,W)` uint8 mask |

**GPU fallback logic:**
```python
available = ort.get_available_providers()
if "CUDAExecutionProvider" in available:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]  # CPU fallback with warning
```

### `src/perception/utils.py`

Visualization utilities for segmentation masks.

| Symbol | Type | Description |
|--------|------|-------------|
| `CLASS_COLORS` | Dict | `{0: (0,255,0), 1: (0,0,255), 2: (255,0,0)}` BGR |
| `colorize_mask(mask) → np.ndarray` | Function | Converts `(H,W)` class mask to `(H,W,3)` BGR image |
| `overlay_mask(frame, mask, alpha=0.4) → np.ndarray` | Function | Alpha-blends colorized mask onto frame |

### `src/mapping/costmap_builder.py`

Builds 2D occupancy grid from segmentation masks and robot poses.

| Symbol | Type | Description |
|--------|------|-------------|
| `_pixel_to_world(u, v, depth, K, cam_height, pitch)` | Function | Back-projects pixel to world XY via ground-plane assumption |
| `build_costmap(masks, poses, K, cam_height, pitch, resolution, grid_size_m, inflate_radius)` | Function | Main costmap builder. Returns `(costmap, origin)` |

**Parameters:**
- `masks`: `(N, H, W)` uint8 — segmentation masks
- `poses`: `(N, 3)` float — robot world positions
- `K`: `(3, 3)` float — camera intrinsic matrix
- `resolution`: meters per grid cell (default 0.05)
- `grid_size_m`: physical grid size in meters (default 80.0)
- `inflate_radius`: obstacle inflation radius in meters (default 0.3)

**Returns:**
- `costmap`: `(grid_n, grid_n)` uint8 — `0`=free, `127`=unknown, `255`=occupied
- `origin`: `(2,)` float — world XY of `costmap[0,0]`

### `src/mapping/viz.py`

Composites visualization overlays onto camera frames.

| Symbol | Type | Description |
|--------|------|-------------|
| `draw_frame(frame, mask, pose, waypoints, costmap, origin, cmd, resolution)` | Function | Creates annotated frame with segmentation overlay, costmap mini-map, pose indicator, and HUD text |

### `src/planning/astar_planner.py`

A* pathfinding on 2D occupancy grid.

| Symbol | Type | Description |
|--------|------|-------------|
| `_heuristic(a, b)` | Function | Euclidean distance between two grid points |
| `astar(costmap, origin, start_xy, goal_xy, resolution)` | Function | Returns list of world `(x,y)` waypoints |

**Behavior:**
- 8-connected grid with diagonal costs = √2
- Occupied cells (`255`) are impassable
- Unknown cells (`127`) are passable
- Falls back to straight line if start/goal blocked or no path found

### `src/planning/pure_pursuit.py`

Pure Pursuit local path-tracking controller.

| Symbol | Type | Description |
|--------|------|-------------|
| `pure_pursuit_step(pose, waypoints, lookahead, max_v, max_w)` | Function | Returns `(v, w)` velocity commands |

**Parameters:**
- `pose`: `(x, y, yaw)` — current robot pose (yaw in radians)
- `waypoints`: list of `(x, y)` — global path from A*
- `lookahead`: target distance in meters (default 1.5)
- `max_v`: max linear velocity in m/s (default 1.5)
- `max_w`: max angular velocity in rad/s (default 1.0)

### `src/slam/orb_slam3_wrapper.py`

Thin wrapper around pyslam's ORB-SLAM3 for monocular visual SLAM.

| Symbol | Type | Description |
|--------|------|-------------|
| `OrbSlam3Mono` | Class | ORB-SLAM3 wrapper |
| `OrbSlam3Mono.__init__(config_path)` | Method | Loads ORB vocabulary and config |
| `OrbSlam3Mono.track(gray_frame, timestamp) → np.ndarray or None` | Method | Processes frame, returns 4×4 pose matrix |
| `OrbSlam3Mono.shutdown()` | Method | Cleans up SLAM system |

### `src/ui/streamlit_app.py`

Interactive web dashboard for exploring pipeline results.

| Symbol | Type | Description |
|--------|------|-------------|
| `load_artifacts()` | Cached Function | Loads masks, poses, costmap, origin, waypoints from disk |

**UI sections:**
- Video player (`st.video`)
- Costmap explorer with trajectory overlay
- Segmentation mask viewer
- Frame selector sidebar

---

## Data Artifacts

All artifacts are saved to `data/rugd/scene_03/`:

| File | Format | Shape / Size | Contents |
|------|--------|-------------|----------|
| `rgb/frame_XXXX.png` | PNG | 1376×1110×3 | Raw camera frames from RUGD |
| `meta.json` | JSON | — | Camera intrinsics and config |
| `masks.npy` | NumPy | `(N, 360, 640)` uint8 | Per-frame segmentation masks |
| `poses.txt` | Text | `(N, 3)` float | Robot world positions (x, y, z) |
| `costmap.npy` | NumPy | `(1600, 1600)` uint8 | 2D occupancy grid |
| `origin.npy` | NumPy | `(2,)` float | World XY of grid[0,0] |
| `waypoints.npy` | NumPy | `(M, 2)` float | A* path waypoints |
| `cmd_vel.npy` | NumPy | `(N, 2)` float | Velocity commands (v, w) |
| `models/fastscnn_rugd.onnx` | ONNX | ~100MB | Exported segmentation model |
| `demo.mp4` | MP4 | 640×360 @ 10fps | Rendered visualization video |

---

## Camera Intrinsics

The RUGD dataset uses a Prosilica GT2750C camera with an 8mm lens mounted on a Clearpath Husky robot at ~25cm ground height. The calibration values used:

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

The pipeline uses GPU acceleration at three stages:

| Stage | Library | GPU Usage | Fallback |
|-------|---------|-----------|----------|
| **ONNX Export** | PyTorch | `model.to('cuda')` for faster export | Falls back to CPU |
| **Segmentation** | ONNX Runtime | `CUDAExecutionProvider` | Falls back to `CPUExecutionProvider` |
| **SLAM** | pyslam | ORB feature extraction on CPU | Always CPU ( pyslam is CPU-only) |

**Detection logic:**
```python
import torch
HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    GPU_NAME = torch.cuda.get_device_name(0)
    _props = torch.cuda.get_device_properties(0)
    GPU_MEM = getattr(_props, 'total_memory', getattr(_props, 'total_mem', 0)) / 1e9
```

**ONNX Runtime provider selection:**
```python
available = ort.get_available_providers()
if "CUDAExecutionProvider" in available:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
```

When no GPU is available, the pipeline runs correctly but segmentation is ~10-50× slower.

---

## Configuration Parameters

### Pipeline Parameters (Notebook Cells)

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `DATA_ROOT` | `data/rugd/scene_03` | Cell 2 | Root directory for all data |
| `MODEL_PATH` | `models/fastscnn_rugd.onnx` | Cell 4 | ONNX model path |
| `input_size` | `(640, 360)` | Cell 5 | Model input resolution (W×H) |
| `resolution` | `0.05` m/cell | Cell 7 | Costmap grid resolution |
| `grid_size_m` | `80.0` m | Cell 7 | Physical size of costmap |
| `inflate_radius` | `0.3` m | Cell 7 | Obstacle safety margin |
| `lookahead` | `1.5` m | Cell 8 | Pure pursuit lookahead distance |
| `max_v` | `1.5` m/s | Cell 8 | Maximum linear velocity |
| `max_w` | `1.0` rad/s | Cell 8 | Maximum angular velocity |
| `goal_distance` | `12.0` m | Cell 8 | How far ahead to plan (X-axis) |
| `video_fps` | `10.0` | Cell 9 | Output video frame rate |
| `TARGET_W × TARGET_H` | `640 × 360` | Cell 9 | Output video resolution |

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
--index-url https://download.pytorch.org/whl/cu118
torch                           # PyTorch (ONNX export, model loading)
torchvision                     # Vision models (DeepLabV3-ResNet50)
torchaudio                      # Audio processing (PyTorch dependency)
onnxruntime-gpu                 # GPU-accelerated ONNX inference
opencv-python-headless          # Image I/O and processing
numpy                           # Array operations
matplotlib                      # Plotting (optional)
tqdm                            # Progress bars
streamlit                       # Web UI framework
pyngrok                         # ngrok tunnel for Colab
gdown                           # Google Drive downloader
```

### System Packages (Colab/Docker)

```
cmake, build-essential          # C++ compilation (pyslam)
libopencv-dev                   # OpenCV headers
wget, unzip                     # Dataset download/extraction
python3.10, python3-pip         # Python runtime
libglib2.0-0, libsm6, libxext6  # OpenCV GUI dependencies
libxrender-dev, libgl1-mesa-glx
```

---

## Docker

### Build

```bash
docker build -t ugv-nav .
```

Uses `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` as base image with CUDA 11.8 + cuDNN 8.

### Run

```bash
docker run --gpus all -p 8888:8888 -v $(pwd)/data:/app/data ugv-nav
```

Opens Jupyter Notebook on `http://localhost:8888`. The `data/` volume mount persists downloaded RUGD data.

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

### Fast-SCNN
```bibtex
@inproceedings{fastscnn,
  title     = {Fast Semantic Segmentation for Autonomous Driving},
  author    = {Poudel, Rudra and Li, Siwen and Bonnetat, Stephan},
  booktitle = {NeurIPS Workshop},
  year      = {2018}
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
