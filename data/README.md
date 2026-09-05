# RUGD Dataset

This folder is **git-ignored** (large files).
Run the notebook to fetch and process the RUGD data (~5.3 GB download).

## Scenes Used

The pipeline processes three terrain sequences from the RUGD dataset:

| Scene | Directory | Frames | Terrain Type |
|-------|-----------|--------|-------------|
| **Creek** | `scene_03/` | ~836 | Rock bed, water, boulders |
| **Village** | `village/` | ~117 | Buildings, paved roads, fences |
| **Trail** | `trail_7/` | ~290 | Forest path, gravel, trees |

## About

The **RUGD** (Robot Unstructured Ground Driving) dataset contains 24 semantic classes across 18 video sequences recorded on a Clearpath Husky robot in unstructured outdoor environments. This project uses three visually distinct sequences to demonstrate multi-terrain autonomous navigation.

## Directory Structure (after download)

```
data/rugd/
├── scene_03/                  # Creek sequence
│   ├── rgb/                   # Extracted frames (frame_0000.png, ...)
│   ├── meta.json              # Camera intrinsics (fx, fy, cx, cy, height, pitch)
│   ├── masks.npy              # Segmentation masks (N, H, W) uint8
│   ├── poses.txt              # Robot poses (N, 3) float [x, y, yaw]
│   ├── costmap.npy            # Gradient occupancy grid (G, G) uint8
│   ├── origin.npy             # Grid origin (2,) float
│   ├── waypoints.npy          # A* path (M, 2) float
│   ├── cmd_vel.npy            # Velocity commands (N, 2) float
│   └── demo.mp4               # Per-scene demo video (H.264)
├── village/                   # Village sequence (same structure)
└── trail_7/                   # Trail sequence (same structure)
```

## Attribution

RUGD dataset is released under **CC-BY-4.0**.
If you use the data, please cite:

```bibtex
@inproceedings{RUGD2019IROS,
  author    = {Wigness, Maggie and Eum, Sungmin and Rogers, John G and Han, David and Kwon, Heesung},
  title     = {A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments},
  booktitle = {International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2019}
}
```

Official website: http://rugd.vision/
