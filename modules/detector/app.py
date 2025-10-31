import os
import cv2
import numpy as np
from PIL import Image
from modules.face_alignment import align

class FaceDetector:
    def __init__(self, device='cuda:0'):
        self.device = device
        print(f"MTCNN model initialized on {device}")

    def detect_faces(self, image_path):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, []

        img = Image.open(image_path).convert('RGB')
        bboxes, faces = align.get_aligned_faces(image_path, rgb_pil_image=img)
        detections = []

        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            det = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'score': float(score)
            }
            detections.append(det)

        # print(f"Detected {len(detections)} faces")
        return img, detections, faces

    def visualize_detections(self, pil_image, detections, output_path=None):
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{det['score']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if output_path:
            cv2.imwrite(output_path, img_cv)
            print(f"Saved visualization to {output_path}")

        return img_cv


if __name__ == "__main__":
    detector = FaceDetector(device='cuda:0')
    img_path = "./face_database/lfw_home/lfw_test/Aaron_Tippin/Aaron_Tippin_0001.jpg"

    pil_img, detections, aligned_faces = detector.detect_faces(img_path)
    print(f"Aligned {len(aligned_faces)} faces")

    os.makedirs("aligned_faces", exist_ok=True)
    for i, face in enumerate(aligned_faces):
        face.save(f"aligned_faces/face_{i}.jpg")

    detector.visualize_detections(pil_img, detections, "detections.jpg")
