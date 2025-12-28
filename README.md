# Real-Time-Face-Recognition-for-Finding-Missing-Person

Set-up guidance for finding missing person in real time using face recognition. This version adds periodic database updates, per-frame batching, temporary crop/embedding storage, alerting, and optional crowd-density monitoring.

## Overview
- Detects faces with YOLOv8-Face
- Processes every 10th frame for efficiency
- Crops faces and saves to `temp_crops/`
- Computes Facenet embeddings and saves to `temp_embeddings/`
- Matches against a periodically refreshed reference index built from images fetched from MongoDB
- Logs matches and triggers alerts via an Alert Manager
- Optionally monitors crowd density and triggers alerts

## Requirements
- Python 3.9+
- A webcam, video file, or RTSP stream
- YOLOv8-Face weights file `yolov8n-face.pt` in the project root (or update the detector to point to your file)
- Dependencies from `requirements.txt`
- Optional: PyTorch with CUDA for GPU acceleration

## Setup
1. Create and activate a virtual environment
   - Windows (PowerShell):
     ```
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Install the correct PyTorch build (optional, recommended for GPU)
   - Follow https://pytorch.org/get-started/locally/ for CUDA-enabled builds
4. Place YOLOv8-Face weights
   - Put `yolov8n-face.pt` in the project root (expected by `src/detection/face2.py`)
5. Configure alerts (optional but recommended)
   - `pipeline_advanced.py` initializes `AlertManager` with a MongoDB URI; update it in code or via your preferred configuration mechanism

## How to Run
- Webcam (index 0)
  ```
  python pipeline_advanced.py --source 0 --conf 0.5 --threshold 0.5 --db-interval 60
  ```
- RTSP stream
  ```
  python pipeline_advanced.py --source rtsp://user:pass@host:554/stream --conf 0.5 --threshold 0.5 --db-interval 60
  ```
- Video file
  ```
  python pipeline_advanced.py --source d:\videos\input.mp4 --conf 0.5 --threshold 0.5 --db-interval 60
  ```
- Final Metric Evaluation
  ```
   python pipeline.py --config configs/default.yaml  
   ```
Flags:
- `--source` webcam index (`0`), a file path, or RTSP URL
- `--conf` face detection confidence threshold
- `--threshold` recognition similarity threshold (score01 in [0..1])
- `--db-interval` seconds between reference DB refreshes



## Tips
- Increase `--conf` to make detection stricter; increase `--threshold` to reduce false positives
- Ensure `yolov8n-face.pt` exists; detection will fail without it
- If GPU is available, the pipeline uses CUDA automatically for embeddings and crowd monitor
- Keep your MongoDB credentials secure; prefer environment-based configuration for production
