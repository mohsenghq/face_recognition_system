# tracker.py
import cv2
from yolox.tracker.byte_tracker import BYTETracker, STrack
import numpy as np

class FaceTracker:
    def __init__(self, fps=30, track_buffer=30):
        self.tracker = BYTETracker(fps=fps, track_buffer=track_buffer)

    def update(self, detections, img_shape):
        """
        detections: bbox و score from RetinaFace
        img_shape: ابعاد تصویر (height, width)
        """
        height, width = img_shape[:2]

        # change format
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det.get('score', 1.0)
            dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets, dtype=np.float32)

        # run tracker
        tracks = self.tracker.update(dets, [height, width], (height, width))

        # outputs
        track_results = []
        for track in tracks:
            bbox = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id
            track_results.append({'track_id': track_id, 'bbox': bbox})
        return track_results
