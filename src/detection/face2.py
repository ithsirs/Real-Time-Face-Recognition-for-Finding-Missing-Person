


# src/detection/face_yolo.py
import cv2
import torch
import os
import sys
from ultralytics import YOLO

class FaceDetector:
    """
    Implements face detection using the YOLOv8-Face model.
    This class is designed to be imported into other scripts.
    """
    def __init__(self, model_path="yolov8n-face.pt", conf_thresh=0.3, device=None):
        self.conf_thresh = conf_thresh
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing YOLOv8-Face Detector on device: {self.device}")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found at: {model_path}", file=sys.stderr)
            sys.exit(1)
            
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame):
        """
        Runs YOLOv8-Face detection on the given BGR frame.
        
        Returns:
            A list of tuples, where each tuple contains:
            (x1, y1, x2, y2, confidence, landmarks_dict)
        """
        results = self.model(frame, verbose=True, imgsz=640)[0]
        
        detections = []
        boxes = results.boxes
        keypoints = results.keypoints

        # Iterate through each detected box
        for i in range(len(boxes)):
            box = boxes[i]
            conf = float(box.conf[0])

            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            landmarks = {}
            # Safely check for corresponding keypoints
            if keypoints is not None and len(keypoints) > i:
                kpts = keypoints[i]
                if len(kpts.xy[0]) == 5:
                    lm_coords = kpts.xy[0].cpu().numpy().astype(int)
                    landmarks = {
                        'left_eye': tuple(lm_coords[0]),
                        'right_eye': tuple(lm_coords[1]),
                        'nose': tuple(lm_coords[2]),
                        'left_mouth': tuple(lm_coords[3]),
                        'right_mouth': tuple(lm_coords[4])
                    }
            
            # Append the detection
            detections.append((x1, y1, x2, y2, conf, landmarks))
                
        return detections
  


