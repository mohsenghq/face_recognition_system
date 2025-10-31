import cv2
import numpy as np
import sys
import argparse
from pathlib import Path
import time
import os
os.environ['INSIGHTFACE_NO_WARNING'] = '1'
os.environ['INSIGHTFACE_LOG_LEVEL'] = 'ERROR'

# Add modules to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from modules.detector.app import FaceDetector
from modules.embedding.app import FaceEmbedder
from modules.matching.app import FaceMatcher
from modules.vector_db.app import FaceVectorDB
from modules.video_handler.app import VideoHandler
from modules.db_monitor.app import DatabaseWatcher

def detect_camera_source():
    """
    Automatically detect available camera source
    Returns the first available camera index or None if no camera found
    """
    print("Detecting camera source...")
    
    # Try different camera indices
    for camera_index in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                print(f"Found camera at index {camera_index}")
                return camera_index
            cap.release()
    
    print("No camera detected")
    return None

def detect_mode(source):
    """
    Automatically detect the processing mode based on source
    """
    if source == "camera" or isinstance(source, int):
        return "camera"
    elif isinstance(source, str) and os.path.exists(source):
        # Check if it's an image file
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if any(source.lower().endswith(ext) for ext in img_extensions):
            return "image"

        # Check if it's a video file
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return "video"
    
    return "camera"  # Default fallback

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


class FaceRecognitionSystem:
    def __init__(self, db_path: str = "face_database", enable_monitoring: bool = True):
        """
        Initialize the complete face recognition system
        
        Args:
            db_path: Path to the face database folder
            enable_monitoring: Whether to enable database monitoring
        """
        self.db_path = db_path
        self.enable_monitoring = enable_monitoring
        
        print("Initializing Face Recognition System...")
        
        # Initialize components
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.vector_db = FaceVectorDB(db_path)
        
        # Initialize matcher with vector database reference
        self.matcher = FaceMatcher(self.vector_db)
        
        # Initialize database watcher
        self.db_watcher = None
        if enable_monitoring:
            self.db_watcher = DatabaseWatcher(db_path, self.on_database_change)
        
        print("System initialized successfully!")
    
    def on_database_change(self):
        """Callback for database changes"""
        print("Database change detected, updating embeddings...")
        self.update_database()
    
    def update_database(self, force_update: bool = False):
        """Update the vector database"""
        self.vector_db.update_database(self.embedder, force_update)
        
        # Update matcher reference (no need to recreate, just update the reference)
        self.matcher.set_vector_db(self.vector_db)
        
        # Get database stats
        stats = self.vector_db.get_database_stats()
        print(f"Database updated: {stats['total_persons']} persons, {stats['total_embeddings']} embeddings")
    
    def start_monitoring(self):
        """Start database monitoring"""
        if self.db_watcher:
            self.db_watcher.start()
    
    def stop_monitoring(self):
        """Stop database monitoring"""
        if self.db_watcher:
            self.db_watcher.stop()
    
    def process_image(self, image_path: str, save_output: bool = True, show_window: bool = False) -> np.ndarray:
        """
        Process a single image for face recognition
        
        Args:
            image_path: Path to input image
            save_output: Whether to save output image
            show_window: Whether to display image in a window
            
        Returns:
            Processed image with annotations
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        processed_img = img.copy()
        
        # Detect faces
        _, detections, _ = self.detector.detect_faces(image_path)
        
        if not detections:
            print("No faces detected in image")
            return processed_img
        
        # Extract face regions and get embeddings
        face_regions = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            face_crop = img[y1:y2, x1:x2]
            face_resized = cv2.resize(face_crop, (112, 112))
            face_regions.append(face_resized)
        
        # Get embeddings and match
        embeddings = self.embedder.get_embeddings(face_regions)
        matches = self.matcher.match(embeddings)
        
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
            img_height, img_width = processed_img.shape[:2]
            thickness = max(2, int(img_width / 400))  # Adaptive thickness
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with adaptive text
            label = f"{person_id} ({similarity:.2f})"
            processed_img = draw_adaptive_text(
                processed_img, 
                label, 
                (x1, y1 - 10), 
                (255, 255, 255),  # White text
                bg_color,  # Background color
                thickness
            )
        
        # Save output if requested
        if save_output:
            output_path = f"output_{Path(image_path).stem}.jpg"
            cv2.imwrite(output_path, processed_img)
            print(f"Processed image saved as: {output_path}")
        
        # Show window if requested
        if show_window:
            window_name = f"Face Recognition - {Path(image_path).name}"
            cv2.imshow(window_name, processed_img)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return processed_img
    
    def process_video(self, source: str = "camera", fps: int = 10, 
                     show_video: bool = True, save_output: bool = True):
        """
        Process video stream for face recognition
        
        Args:
            source: Video source ("camera" or path to video file)
            fps: Processing FPS
            show_video: Whether to display video window
            save_output: Whether to save output video
        """
        try:
            video_handler = VideoHandler(source, fps)
            video_handler.process_video(
                self.detector, 
                self.embedder, 
                self.matcher,
                show_video=show_video,
                save_output=save_output
            )
        except Exception as e:
            print(f"Error processing video: {e}")
    
    def add_person(self, person_id: str, name: str, image_paths: list):
        """Add a new person to the database"""
        success = self.vector_db.add_person(person_id, name, image_paths, self.embedder)
        if success:
            self.update_database()
            print(f"Successfully added person: {name}")
        else:
            print(f"Failed to add person: {name}")
    
    def get_database_stats(self):
        """Get database statistics"""
        return self.vector_db.get_database_stats()

def main(args):
    
    # Auto-detect source if not provided
    if args.source is None:
        camera_index = detect_camera_source()
        if camera_index is not None:
            source = camera_index
            print(f"Using auto-detected camera at index {camera_index}")
        else:
            print("No camera found. Please specify a source with --source")
            return
    else:
        source = args.source
        # Try to convert to int if it's a number (camera index)
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string for file paths
    
    # Auto-detect mode based on source
    mode = detect_mode(source)
    print(f"Auto-detected mode: {mode}")
    
    # Initialize system
    system = FaceRecognitionSystem(
        db_path=args.db_path,
        enable_monitoring=not args.no_monitoring
    )
    
    # Update database on startup
    print("Updating database...")
    system.update_database(force_update=args.update_db)
    
    # Start monitoring
    if not args.no_monitoring:
        system.start_monitoring()
    
    try:
        if mode == "image":
            print(f"Processing image: {source}")
            system.process_image(source, save_output=True, show_window=args.show)
            
        elif mode == "video":
            print(f"Processing video: {source}")
            system.process_video(
                source=source,
                fps=args.fps,
                show_video=True,
                save_output=True
            )
            
        elif mode == "camera":
            print(f"Processing camera feed from index {source}")
            system.process_video(
                source=source,
                fps=args.fps,
                show_video=True,
                save_output=False
            )
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Stop monitoring
        if not args.no_monitoring:
            system.stop_monitoring()
        
        print("System shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--source", default='./test.mp4', 
                       help="Input source (camera index, video file path, or image file path). Auto-detects camera if not specified.")
    parser.add_argument("--db-path", default="face_database/lfw_home/lfw_funneled", 
                       help="Path to face database folder")
    parser.add_argument("--fps", type=int, default=10, 
                       help="Processing FPS for video")
    parser.add_argument("--no-monitoring", action="store_true", 
                       help="Disable database monitoring")
    parser.add_argument("--update-db", action="store_true", 
                       help="Force database update on startup")
    parser.add_argument("--show", default='true',
                       help="Show result in a window (for image mode)",
                       type=lambda x: x.lower() == 'true')
    
    args = parser.parse_args()
    main(args)
