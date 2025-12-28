# Face Detection, Tracking & Recognition System

This project implements an **end-to-end real-time face detection, tracking, and recognition pipeline** using YOLOv8-Face, DeepSORT, and FaceNet (InceptionResnetV1).

---

## 1. System Overview

### Core Capabilities

* Real-time face detection with landmarks
* Persistent face ID tracking across frames
* Face crop extraction from live video
* Face embedding generation (512-D)
* Offline embedding database creation
* Face similarity search using cosine similarity

### Technology Stack

* **Detection**: YOLOv8-Face (Ultralytics)
* **Tracking**: DeepSORT (deep-sort-realtime)
* **Recognition**: FaceNet (InceptionResnetV1 – facenet-pytorch)
* **Backend**: Python, OpenCV, PyTorch

---

## 2. Project Directory Structure

```
project_root/
│
├── src/
│   ├── detection/
│   │   ├── face2.py
│   │   └── test1_dt.py
│   ├── tracking/
│   │   └── deep_sort.py
│   └── recognition/
│       ├── face_recog_core.py
│       ├── precompute_embeddings.py
│       └── search_query.py
│
├── embeddings/               # Generated face embeddings
├── models/                   # Model files (optional)
├── runs/                     # Logs
├── .save/                    # Saved face crops from live video
└── yolov8n-face.pt           # YOLOv8 face detection model
```

---

## 3. Environment Setup (Step-by-Step)

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

---

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python ultralytics facenet-pytorch
pip install deep-sort-realtime scikit-learn pillow numpy
```

> ⚠️ For CUDA support, ensure compatible PyTorch + NVIDIA drivers are installed.

---

### Step 3: Download YOLOv8 Face Model

Download **YOLOv8n-Face** and place it in the project root:

```
yolov8n-face.pt
```

---

## 4. Running the System

### 4.1 Real-Time Face Detection + Tracking (Webcam)

This runs YOLOv8 face detection + DeepSORT tracking and saves face crops.

```bash
python -m src.detection.test1_dt 
```

#### What Happens Internally

1. Webcam frame captured
2. Faces detected with landmarks
3. DeepSORT assigns persistent IDs
4. Bounding boxes + IDs rendered
5. Face crops saved to `.save/`

Press **`q`** to exit.

---

## 5. Face Recognition Pipeline

### 5.1 Prepare Face Dataset

Using the real-time detected faces saved in .save/ as a dataset.

```
.save/
├── 1_20251227_215952.jpg
├── 1_20251227_215953.jpg
├── 1_20251227_215954.jpg
```

---

### 5.2 Precompute Face Embeddings

```bash
python -m src.recognition.precompute_embeddings --dataset_dir .save --precompute  --device cuda
```

#### Output

* `embeddings/*.npy`
* `embeddings/embeddings_index.csv`

---

### 5.3 Search / Identify a Face

```bash
python -m src.recognition.search_query --dataset_dir .save  --query sample3.jpg --topk 3  --show_image
```

#### Output

* Console similarity scores
* Saved result montage (`result_matches.jpg`)

---

## 6. End-to-End Logical Flow

```
Webcam Frame
   ↓
YOLOv8 Face Detection
   ↓
DeepSORT Tracking (Face ID)
   ↓
Face Crop Extraction
   ↓
FaceNet Embedding (512-D)
   ↓
Cosine Similarity Matching
```

---

## 7. Configuration Tips

* **Detection confidence**: `conf_thresh` in `face2.py`
* **Tracking stability**: `max_age`, `n_init` in `deep_sort.py`
* **Recognition threshold**: `--threshold` in `search_query.py`
* **Embedding model**: `vggface2` or `casia-webface`

---

## 8. Common Issues & Fixes

| Issue              | Fix                        |
| ------------------ | -------------------------- |
| Webcam not opening | Check camera index (0/1)   |
| Slow inference     | Use GPU, reduce resolution |
| False matches      | Increase cosine threshold  |
| ID switching       | Increase `n_init`          |

---

## 9. Future Extensions

* Live recognition with tracker IDs
* Database-backed embedding storage (PostgreSQL + pgvector)
* Multi-camera support
* Face re-identification across sessions
* REST / WebSocket API integration

---

## 10. Intended Use Cases

* Missing person identification
* Surveillance analytics
* Smart attendance systems
* Research & academic projects

---

**Author:** Srishti Majumdar
**Domain:** Computer Vision · AI · Face Recognition
