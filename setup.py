#!/usr/bin/env python3
"""
Setup script for Face Recognition System
This script helps set up the face database structure and initial configuration
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np

def create_database_structure(db_path: str = "face_database"):
    """Create the face database folder structure"""
    db_path = Path(db_path)
    
    print(f"Creating database structure at: {db_path}")
    
    # Create main database directory
    db_path.mkdir(exist_ok=True)
    
    # Create example person directories
    example_persons = ["person_1", "person_2", "person_3"]
    
    for person in example_persons:
        person_dir = db_path / person
        person_dir.mkdir(exist_ok=True)
        
        # Create README file for each person
        readme_path = person_dir / "README.txt"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"""Person Database: {person}

Instructions:
1. Add images of {person} to this folder
2. Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
3. Each image should contain clear face(s) of {person}
4. The system will automatically detect and process faces
5. Multiple images per person are recommended for better accuracy

Folder structure:
{person}/
- image1.jpg
- image2.jpg
- image3.jpg
- README.txt
""")
    
    print("Database structure created successfully!")
    print(f"Add images to the person folders in: {db_path}")
    
    return db_path

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def test_system():
    """Test if the system can be imported and initialized"""
    print("Testing system components...")
    
    try:
        # Test imports
        from modules.detector.app import FaceDetector
        from modules.embedding.app import FaceEmbedder
        from modules.matching.app import FaceMatcher
        from modules.vector_db.app import FaceVectorDB
        from modules.video_handler.app import VideoHandler
        
        print("[OK] All modules imported successfully")
        
        # Test basic initialization
        detector = FaceDetector()
        print("[OK] FaceDetector initialized")
        
        embedder = FaceEmbedder()
        print("[OK] FaceEmbedder initialized")
        
        matcher = FaceMatcher(np.array([]), [])
        print("[OK] FaceMatcher initialized")
        
        vector_db = FaceVectorDB("face_database")
        print("[OK] FaceVectorDB initialized")
        
        print("[OK] System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] System test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Face Recognition System")
    parser.add_argument("--db-path", default="face_database", 
                       help="Path for face database folder")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install dependencies")
    parser.add_argument("--test", action="store_true", 
                       help="Test system components")
    parser.add_argument("--all", action="store_true", 
                       help="Run all setup steps")
    
    args = parser.parse_args()
    
    print("Face Recognition System Setup")
    print("=" * 40)
    
    success = True
    
    if args.install_deps or args.all:
        success &= install_dependencies()
        print()
    
    if args.test or args.all:
        success &= test_system()
        print()
    
    # Always create database structure
    create_database_structure(args.db_path)
    print()
    
    if success:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add images to the person folders in face_database/")
        print("2. Run: python main_video.py --mode camera")
        print("3. Or run: python main_video.py --mode image --input your_image.jpg")
    else:
        print("Setup completed with some issues. Please check the errors above.")

if __name__ == "__main__":
    main()
