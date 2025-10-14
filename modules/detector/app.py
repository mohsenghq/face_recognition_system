# detector.py
import cv2
import torch
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.detector = RetinaFace(quality='normal')

    def detect_faces(self, image_path):
        img = cv2.imread(image_path)
        faces = self.detector.predict(img)
        detections = []
        for face in faces:
            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'landmarks': face['landmarks']
            })
        return img, detections
