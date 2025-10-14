# super_resolution.py
import torch
from gfpgan import GFPGANer
import cv2

class FaceEnhancer:
    def __init__(self, model_path='weights/GFPGANv1.4.pth', upscale=2, device='cuda'):
        self.enhancer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )

    def enhance_faces(self, img, detections):
        enhanced_faces = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            face_crop = img[y1:y2, x1:x2]
            _, _, restored = self.enhancer.enhance(face_crop, has_aligned=False, only_center_face=False, paste_back=False)
            enhanced_faces.append(restored)
        return enhanced_faces
