import numpy as np
from cjm_byte_track.core import BYTETracker

class FaceTracker:
    def __init__(self, fps=30, track_buffer=30):
        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=track_buffer,
            match_thresh=0.8,
            frame_rate=fps
        )

    def update(self, face_detections, img_shape):
        """
        face_detections: bbox و score 
            example:
            [
                {"bbox": [x1, y1, x2, y2], "score": 0.98, "id": "person_A"},
                {"bbox": [x1, y1, x2, y2], "score": 0.95, "id": "person_B"},
            ]
        img_shape: (height, width)
        """
        height, width = img_shape[:2]

        if not face_detections:
            return []

        # input of ByteTracker
        dets = []
        ids = []
        for det in face_detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det.get("score", 1.0)
            dets.append([x1, y1, x2, y2, float(score)])
            ids.append(det.get("id"))  # id matching

        dets = np.array(dets, dtype=np.float32)

        # tracking
        tracks = self.tracker.update(dets)

        # matching
        track_results = []
        for i, track in enumerate(tracks):
            bbox = track.tlbr
            track_id = track.track_id
            person_id = ids[i] if i < len(ids) else None
            track_results.append({
                "track_id": track_id,
                "bbox": bbox,
                "person_id": person_id
            })

        return track_results
