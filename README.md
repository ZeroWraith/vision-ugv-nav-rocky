# Vision‑Based Autonomous Navigation for UGV (Outdoor, GPS‑Denied)

**Hackathon prototype** – runs completely in Google Colab (free T4 GPU) on the **RUGD rocky** sequence.

## Quick‑start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerowraith/vision-ugv-nav-rocky/blob/main/notebooks/01_full_pipeline.ipynb)

1. Click the badge above.  
2. Runtime → **Change runtime type** → **GPU (T4)**.  
3. Run **all cells** (≈ 4 min).  
4. At the end a public **ngrok** URL appears – open it for the interactive Streamlit UI.  
5. `demo.mp4` is also saved in the notebook output and can be downloaded.

## Repository layout
```
vision-ugv-nav-rocky/
├─ data/                     # downloaded RUGD rocky sequence (git‑ignored)
├─ notebooks/
│   └─ 01_full_pipeline.ipynb
├─ src/
│   ├─ perception/          # Fast‑SCNN ONNX inference
│   ├─ slam/                # ORB‑SLAM3 mono wrapper (pyslam)
│   ├─ mapping/             # Cost‑map builder
│   ├─ planning/            # A* + Pure Pursuit
│   └─ ui/                  # Streamlit dashboard
├─ models/
│   └─ fastscnn_rugd.onnx   # pre‑exported segmentation model
├─ scripts/
│   └─ download_rugd.sh     # pulls the rocky sequence
├─ requirements.txt
├─ Dockerfile               # optional reproducible image
├─ LICENSE                  # MIT
└─ README.md
```

## Pipeline overview
| Block | Method | Key library |
|-------|--------|-------------|
| **Perception** | Fast‑SCNN semantic segmentation (traversable / obstacle / sky) | ONNX‑Runtime (GPU) |
| **Visual Localization** | ORB‑SLAM3 monocular | pyslam (pre‑built wheel) |
| **Mapping** | Project traversable pixels to world XY using SLAM poses → 2‑D occupancy grid + inflation | NumPy / OpenCV |
| **Global Planner** | A* on the cost‑map | Custom Python |
| **Local Controller** | Pure Pursuit (bicycle model) | Custom Python |
| **Visualization / UI** | Annotated video + Streamlit (ngrok tunnel) | Streamlit, ngrok |

## License
MIT – see `LICENSE`. RUGD dataset is CC‑BY‑4.0; attribution included in `data/README.md`.

## Citation
If you use this code, please cite the RUGD dataset and the original Fast‑SCNN / ORB‑SLAM3 papers.