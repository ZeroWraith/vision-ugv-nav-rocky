# RUGD Dataset (Creek Sequence)

This folder is **git-ignored** (large files).
Run `scripts/download_rugd.sh` or execute the notebook to fetch the *creek* subsequence (~2000+ frames of rocky terrain).

## About

The **creek** sequence from the RUGD dataset contains areas near a body of water with significant rock-bed terrain — ideal for testing off-road navigation in unstructured environments.

## Directory Structure (after download)

```
data/rugd/scene_03/
├── rgb/              # Extracted creek frames (frame_0000.png, ...)
├── meta.json         # Camera intrinsics (fx, fy, cx, cy, height, pitch)
├── masks.npy         # Segmentation masks (N, H, W) uint8
├── poses.txt         # Robot poses (N, 3) float
├── costmap.npy       # Occupancy grid (G, G) uint8
├── origin.npy        # Grid origin (2,) float
├── waypoints.npy     # A* path (M, 2) float
└── cmd_vel.npy       # Velocity commands (N, 2) float
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
