#!/usr/bin/env python3
"""
Demo script for Face Recognition System
This script demonstrates how to use the face recognition system
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add modules to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from modules.detector.app import FaceDetector
from modules.embedding.app import FaceEmbedder
from modules.matching.app import FaceMatcher
from modules.vector_db.app import FaceVectorDB

def demo_basic_usage():
    """Demonstrate basic usage of the face recognition system"""
    print("=== Face Recognition System Demo ===")
    
    # Initialize components
    print("1. Initializing components...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    vector_db = FaceVectorDB("face_database")
    
    # Update database
    print("2. Updating database...")
    vector_db.update_database(embedder)
    
    # Get database stats
    stats = vector_db.get_database_stats()
    print(f"3. Database contains {stats['total_persons']} persons with {stats['total_embeddings']} embeddings")
    print(f"   Persons: {', '.join(stats['persons'])}")
    
    # Get embeddings for matching
    embeddings, labels = vector_db.get_all_embeddings()
    matcher = FaceMatcher(embeddings, labels)
    
    print("4. System ready for face recognition!")
    print("\nTo use the system:")
    print("- Camera: python main_video.py --mode camera")
    print("- Video file: python main_video.py --mode video --source path/to/video.mp4")
    print("- Single image: python main_video.py --mode image --input path/to/image.jpg")

def demo_database_management():
    """Demonstrate database management features"""
    print("\n=== Database Management Demo ===")
    
    vector_db = FaceVectorDB("face_database")
    
    # Show database structure
    print("1. Current database structure:")
    person_images = vector_db.scan_database_folder()
    for person_id, image_paths in person_images.items():
        print(f"   {person_id}: {len(image_paths)} images")
        for img_path in image_paths[:2]:  # Show first 2 images
            print(f"     - {Path(img_path).name}")
        if len(image_paths) > 2:
            print(f"     - ... and {len(image_paths) - 2} more")
    
    # Show person metadata
    print("\n2. Person metadata:")
    for person_id in person_images.keys():
        info = vector_db.get_person_info(person_id)
        print(f"   {person_id}:")
        print(f"     - Name: {info.get('name', 'N/A')}")
        print(f"     - Images: {info.get('total_images', 0)}")
        print(f"     - Last updated: {info.get('last_updated', 'N/A')}")

def demo_adding_person():
    """Demonstrate how to add a new person"""
    print("\n=== Adding New Person Demo ===")
    print("To add a new person to the database:")
    print("1. Create a folder in face_database/ with the person's name")
    print("2. Add multiple clear face images to that folder")
    print("3. The system will automatically detect and process them")
    print("\nExample:")
    print("face_database/")
    print("└── John_Doe/")
    print("    ├── photo1.jpg")
    print("    ├── photo2.jpg")
    print("    └── photo3.jpg")
    print("\nThe system will automatically:")
    print("- Detect faces in each image")
    print("- Extract embeddings")
    print("- Add them to the vector database")
    print("- Update the face matcher")

def main():
    """Run all demos"""
    try:
        demo_basic_usage()
        demo_database_management()
        demo_adding_person()
        
        print("\n=== Demo Complete ===")
        print("The face recognition system is ready to use!")
        print("Check the README.md file for detailed usage instructions.")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure all dependencies are installed and the system is properly set up.")

if __name__ == "__main__":
    main()
