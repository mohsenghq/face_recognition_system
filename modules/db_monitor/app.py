# db_monitor.py
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, Optional

class DatabaseMonitor(FileSystemEventHandler):
    """Monitor database folder for changes and trigger updates"""
    
    def __init__(self, db_path: str, on_change: Optional[Callable] = None):
        """
        Initialize database monitor
        
        Args:
            db_path: Path to monitor
            on_change: Callback function when changes are detected
        """
        self.db_path = Path(db_path)
        self.on_change = on_change
        self.last_modified = {}
        
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self._is_image_file(event.src_path):
            print(f"New image detected: {event.src_path}")
            self._trigger_update()
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self._is_image_file(event.src_path):
            # Avoid duplicate triggers for the same file
            current_time = time.time()
            if event.src_path not in self.last_modified or \
               current_time - self.last_modified[event.src_path] > 1.0:
                print(f"Image modified: {event.src_path}")
                self.last_modified[event.src_path] = current_time
                self._trigger_update()
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory and self._is_image_file(event.src_path):
            print(f"Image deleted: {event.src_path}")
            self._trigger_update()
    
    def on_moved(self, event):
        """Handle file move/rename"""
        if not event.is_directory and self._is_image_file(event.dest_path):
            print(f"Image moved: {event.src_path} -> {event.dest_path}")
            self._trigger_update()
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions
    
    def _trigger_update(self):
        """Trigger database update"""
        if self.on_change:
            self.on_change()

class DatabaseWatcher:
    """Watch database folder for changes"""
    
    def __init__(self, db_path: str, on_change: Optional[Callable] = None):
        """
        Initialize database watcher
        
        Args:
            db_path: Path to watch
            on_change: Callback function when changes are detected
        """
        self.db_path = Path(db_path)
        self.on_change = on_change
        self.observer = Observer()
        self.event_handler = DatabaseMonitor(str(self.db_path), on_change)
        
    def start(self):
        """Start watching the database folder"""
        if not self.db_path.exists():
            print(f"Creating database directory: {self.db_path}")
            self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.observer.schedule(self.event_handler, str(self.db_path), recursive=True)
        self.observer.start()
        print(f"Started watching database folder: {self.db_path}")
    
    def stop(self):
        """Stop watching the database folder"""
        self.observer.stop()
        self.observer.join()
        print("Stopped watching database folder")
    
    def is_running(self) -> bool:
        """Check if watcher is running"""
        return self.observer.is_alive()
