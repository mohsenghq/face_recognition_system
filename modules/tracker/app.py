import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

class FaceTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        """
        ByteTrack tracker initialization.
        Parameters:
            track_thresh: confidence threshold for detections to be considered
            match_thresh: threshold for matching tracks
            track_buffer: number of frames to keep lost tracks
            frame_rate: video frame rate
        """
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate
        )

    def update(self, detections, img_info):
        """
        Update the tracker with new detections from the detector.
        Parameters:
            detections: list or np.array of [x1, y1, x2, y2, score]
            img_info: dict containing at least 'height' and 'width'
        Returns:
            List of dictionaries with each track info:
            [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ]
        """
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(detections)

        online_targets = self.tracker.update(
            output_results=dets,
            img_info=[img_info['height'], img_info['width']],
            img_size=[img_info['height'], img_info['width']]
        )

        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            tracks.append({
                "track_id": track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        return tracks
