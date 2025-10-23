# detector.py
import cv2
import torch
from retinaface.RetinaFace import detect_faces

class FaceDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.threshold = 0.9  # Detection threshold

    def detect_faces(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, []
        
        # Use retinaface detect_faces function
        faces = detect_faces(image_path, threshold=self.threshold)
        
        detections = []
        for face_key, face_data in faces.items():
            facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
            landmarks = face_data['landmarks']
            score = face_data['score']
            
            detections.append({
                'bbox': tuple(facial_area),
                'landmarks': landmarks,
                'score': score
            })
        
        return img, detections
