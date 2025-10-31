# video_handler.py
import cv2
import numpy as np
from typing import Optional, Callable, Tuple
import time

def draw_adaptive_text(img, text, position, color, bg_color=None, thickness=2):
    """
    Draw text with adaptive size and safe positioning.
    Ensures text stays inside the image.
    """
    img_height, img_width = img.shape[:2]
    font_scale = max(0.4, min(2.0, img_width / 800))  # Adaptive font scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size
    x, y = position
    padding = 5

    # Default bottom-left anchor correction
    x = max(padding, min(x, img_width - text_w - padding))
    y = max(text_h + padding, min(y, img_height - padding))

    # Check if text would overflow bottom edge
    if y + text_h + padding > img_height:
        y = img_height - text_h - padding

    # Check if text would overflow right edge
    if x + text_w + padding > img_width:
        x = img_width - text_w - padding

    # Draw background if needed
    if bg_color is not None:
        cv2.rectangle(
            img,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding // 2),
            bg_color,
            -1
        )

    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

class VideoHandler:
    def __init__(self, source: str = "camera", fps: int = 30):
        """
        Initialize video handler
        
        Args:
            source: Video source - "camera" for webcam, camera index (int), or path to video file
            fps: Target FPS for processing
        """
        self.source = source
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        
        # Initialize video capture
        if source == "camera":
            self.cap = cv2.VideoCapture(0)  # Default camera
        elif isinstance(source, int):
            self.cap = cv2.VideoCapture(source)  # Specific camera index
        else:
            self.cap = cv2.VideoCapture(source)  # Video file path
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video source initialized: {self.width}x{self.height} @ {self.video_fps:.2f} FPS")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the video source"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def should_process_frame(self) -> bool:
        """Check if enough time has passed to process next frame"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False
    
    def process_video(self, 
                     detector, 
                     embedder, 
                     matcher, 
                     on_frame_processed: Optional[Callable] = None,
                     show_video: bool = True,
                     save_output: bool = True,
                     output_path: str = "output_video.avi"):
        """
        Process video stream with face recognition
        
        Args:
            detector: FaceDetector instance
            embedder: FaceEmbedder instance  
            matcher: FaceMatcher instance
            on_frame_processed: Callback function for processed frames
            show_video: Whether to display video window
            save_output: Whether to save output video
            output_path: Path for output video file
        """
        # Setup video writer if saving
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        processed_frames = 0
        
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    break
                
                frame_count += 1
                
                # Process frame at specified FPS
                if self.should_process_frame():
                    processed_frame = self.process_single_frame(
                        frame, detector, embedder, matcher
                    )
                    processed_frames += 1
                    
                    # Call callback if provided
                    if on_frame_processed:
                        on_frame_processed(processed_frame, frame_count)
                    
                    # Save frame if requested
                    if save_output and video_writer:
                        video_writer.write(processed_frame)
                    
                    # Show video if requested
                    if show_video:
                        cv2.imshow('Face Recognition', processed_frame)
                        
                        # Check for exit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            break
                else:
                    # Still show video even if not processing
                    if show_video:
                        cv2.imshow('Face Recognition', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()
            
            print(f"Processing complete: {processed_frames}/{frame_count} frames processed")
    
    def process_single_frame(self, frame, detector, embedder, matcher) -> np.ndarray:
        """
        Process a single frame for face recognition
        
        Args:
            frame: Input frame
            detector: FaceDetector instance
            embedder: FaceEmbedder instance
            matcher: FaceMatcher instance
            
        Returns:
            Processed frame with annotations
        """
        processed_frame = frame.copy()
        
        try:
            # Save frame temporarily for detection
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect faces
            _, detections, _ = detector.detect_faces(temp_path)
            
            if detections:
                # Extract face regions
                face_regions = []
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    face_crop = frame[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_crop, (112, 112))
                    face_regions.append(face_resized)
                
                # Get embeddings
                embeddings = embedder.get_embeddings(face_regions)
                
                # Match faces
                matches = matcher.match(embeddings)
                
                # Draw results
                for i, (det, (person_id, similarity)) in enumerate(zip(detections, matches)):
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    
                    # Choose color based on match
                    if person_id != "Unknown":
                        color = (0, 255, 0)  # Green for known person
                        bg_color = (0, 200, 0)  # Darker green background
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        bg_color = (0, 0, 200)  # Darker red background
                    
                    # Draw bounding box with adaptive thickness
                    img_height, img_width = processed_frame.shape[:2]
                    thickness = max(2, int(img_width / 400))  # Adaptive thickness
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label with adaptive text
                    label = f"{person_id} ({similarity:.2f})"
                    processed_frame = draw_adaptive_text(
                        processed_frame, 
                        label, 
                        (x1, y1 - 10), 
                        (255, 255, 255),  # White text
                        bg_color,  # Background color
                        thickness
                    )
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return processed_frame
    
    def release(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.release()
