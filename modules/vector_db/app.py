# vector_db.py
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from datetime import datetime
import hashlib
import faiss

class FaceVectorDB:
    def __init__(self, db_path: str = "face_database"):
        """
        Initialize the face vector database with FAISS
        
        Args:
            db_path: Path to the folder containing person images
        """
        self.db_path = Path(db_path)
        self.vector_file = self.db_path / "face_vectors.pkl"
        self.metadata_file = self.db_path / "face_metadata.json"
        self.faiss_index_file = self.db_path / "faiss_index.bin"
        
        # Create database directory if it doesn't exist
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize storage
        self.face_vectors = {}  # {person_id: [embeddings]} - kept for compatibility
        self.image_embeddings = {}  # {image_path: embedding} for individual image tracking
        self.person_metadata = {}  # {person_id: {name, image_paths, last_updated}}
        self.image_hashes = {}  # {image_path: hash} for change detection
        
        # FAISS index for efficient similarity search
        self.faiss_index = None
        self.embedding_dim = 512  # Default embedding dimension (InsightFace)
        self.id_to_person = {}  # Maps FAISS index ID to person_id
        self.person_to_ids = {}  # Maps person_id to list of FAISS index IDs
        
        # Load existing data
        self.load_database()
        
    def load_database(self):
        """Load existing vector database and metadata"""
        try:
            if self.vector_file.exists():
                with open(self.vector_file, 'rb') as f:
                    data = pickle.load(f)
                    # Handle both old and new format
                    if isinstance(data, dict) and 'image_embeddings' in data:
                        self.face_vectors = data.get('face_vectors', {})
                        self.image_embeddings = data.get('image_embeddings', {})
                    else:
                        # Old format - convert to new format
                        self.face_vectors = data
                        self.image_embeddings = {}
                print(f"Loaded {len(self.face_vectors)} persons from vector database")
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.person_metadata = metadata.get('person_metadata', {})
                    self.image_hashes = metadata.get('image_hashes', {})
                    self.id_to_person = metadata.get('id_to_person', {})
                    self.person_to_ids = metadata.get('person_to_ids', {})
                print(f"Loaded metadata for {len(self.person_metadata)} persons")
            
            # Load FAISS index
            if self.faiss_index_file.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Initialize empty FAISS index
                self._initialize_faiss_index()
                
        except Exception as e:
            print(f"Error loading database: {e}")
            self.face_vectors = {}
            self.image_embeddings = {}
            self.person_metadata = {}
            self.image_hashes = {}
            self.id_to_person = {}
            self.person_to_ids = {}
            self._initialize_faiss_index()
    
    def _initialize_faiss_index(self):
        """Initialize empty FAISS index"""
        # Use IndexFlatIP for inner product similarity (cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        print("Initialized empty FAISS index")
    
    def save_database(self):
        """Save vector database and metadata to disk"""
        try:
            with open(self.vector_file, 'wb') as f:
                pickle.dump({
                    'face_vectors': self.face_vectors,
                    'image_embeddings': self.image_embeddings
                }, f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'person_metadata': self.person_metadata,
                    'image_hashes': self.image_hashes,
                    'id_to_person': self.id_to_person,
                    'person_to_ids': self.person_to_ids
                }, f, indent=2)
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
                
            print("Database saved successfully")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash of image file for change detection"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def scan_database_folder(self) -> Dict[str, List[str]]:
        """Scan the database folder for person images"""
        person_images = {}
        
        if not self.db_path.exists():
            return person_images
            
        for person_folder in self.db_path.iterdir():
            if person_folder.is_dir():
                person_id = person_folder.name
                image_files = []
                
                # Look for common image extensions
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                
                for image_file in person_folder.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        image_files.append(str(image_file))
                
                if image_files:
                    person_images[person_id] = image_files
                    
        return person_images
    
    def _add_embedding_to_faiss(self, embedding: np.ndarray, person_id: str) -> int:
        """Add a single embedding to FAISS index"""
        # Normalize embedding for cosine similarity
        embedding_normalized = embedding / np.linalg.norm(embedding)
        embedding_normalized = embedding_normalized.reshape(1, -1).astype('float32')
        
        # Add to FAISS index
        faiss_id = self.faiss_index.ntotal
        self.faiss_index.add(embedding_normalized)
        
        # Update mappings
        self.id_to_person[faiss_id] = person_id
        if person_id not in self.person_to_ids:
            self.person_to_ids[person_id] = []
        self.person_to_ids[person_id].append(faiss_id)
        
        return faiss_id
    
    def _remove_embeddings_from_faiss(self, person_id: str):
        """Remove all embeddings for a person from FAISS index"""
        if person_id not in self.person_to_ids:
            return
        
        # Get IDs to remove
        ids_to_remove = self.person_to_ids[person_id]
        
        # Create new index without these embeddings
        if self.faiss_index.ntotal > len(ids_to_remove):
            # Create new index with remaining embeddings
            remaining_embeddings = []
            new_id_to_person = {}
            new_person_to_ids = {}
            
            for i in range(self.faiss_index.ntotal):
                if i not in ids_to_remove:
                    # Get embedding from current index
                    embedding = self.faiss_index.reconstruct(i)
                    remaining_embeddings.append(embedding)
                    
                    # Update mappings
                    person = self.id_to_person[i]
                    new_id = len(remaining_embeddings) - 1
                    new_id_to_person[new_id] = person
                    
                    if person not in new_person_to_ids:
                        new_person_to_ids[person] = []
                    new_person_to_ids[person].append(new_id)
            
            # Rebuild index
            if remaining_embeddings:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index.add(np.array(remaining_embeddings).astype('float32'))
            else:
                self._initialize_faiss_index()
            
            # Update mappings
            self.id_to_person = new_id_to_person
            self.person_to_ids = new_person_to_ids
        else:
            # Remove all embeddings
            self._initialize_faiss_index()
            self.id_to_person = {}
            self.person_to_ids = {}
    
    def update_database(self, embedder, force_update: bool = False):
        """
        Update the vector database with new/changed images
        Only processes images that don't have embeddings yet
        
        Args:
            embedder: FaceEmbedder instance
            force_update: Force update even if images haven't changed
        """
        print("Scanning database folder for updates...")
        person_images = self.scan_database_folder()
        
        updated_count = 0
        new_persons = 0
        new_images_processed = 0
        
        for person_id, image_paths in person_images.items():
            # Check if person exists in metadata
            if person_id not in self.person_metadata:
                self.person_metadata[person_id] = {
                    'name': person_id,
                    'image_paths': [],
                    'last_updated': None,
                    'total_images': 0
                }
                new_persons += 1
            
            # Find images that need processing
            images_to_process = []
            current_paths = set(self.person_metadata[person_id]['image_paths'])
            new_paths = set(image_paths)
            
            # Check for new images
            for image_path in image_paths:
                if image_path not in self.image_embeddings:
                    images_to_process.append(image_path)
                elif force_update:
                    # Check if image has changed (for force update)
                    current_hash = self.image_hashes.get(image_path, "")
                    new_hash = self.calculate_image_hash(image_path)
                    if current_hash != new_hash:
                        images_to_process.append(image_path)
            
            if images_to_process:
                print(f"Processing {len(images_to_process)} new/changed images for person: {person_id}")
                person_embeddings = []
                
                for image_path in images_to_process:
                    try:
                        # Load and process image
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"Could not load image: {image_path}")
                            continue
                        
                        # Detect faces in the image
                        from modules.detector.app import FaceDetector
                        detector = FaceDetector()
                        _, detections = detector.detect_faces(image_path)
                        
                        if not detections:
                            print(f"No faces detected in: {image_path}")
                            continue
                        
                        # Get embeddings for all faces (take the largest face)
                        face_embeddings = embedder.get_embeddings([
                            cv2.resize(img[det['bbox'][1]:det['bbox'][3], det['bbox'][0]:det['bbox'][2]], (112, 112))
                            for det in detections
                        ])
                        
                        # Use the face with highest confidence
                        best_face_idx = np.argmax([det.get('score', 0) for det in detections])
                        embedding = face_embeddings[best_face_idx]
                        
                        # Store individual image embedding
                        self.image_embeddings[image_path] = embedding
                        person_embeddings.append(embedding)
                        
                        # Add to FAISS index
                        self._add_embedding_to_faiss(embedding, person_id)
                        
                        # Update image hash
                        self.image_hashes[image_path] = self.calculate_image_hash(image_path)
                        new_images_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue
                
                if person_embeddings:
                    # Rebuild person's embeddings from all their images
                    all_person_embeddings = []
                    for path in image_paths:
                        if path in self.image_embeddings:
                            all_person_embeddings.append(self.image_embeddings[path])
                    
                    if all_person_embeddings:
                        self.face_vectors[person_id] = np.array(all_person_embeddings)
                        self.person_metadata[person_id]['image_paths'] = image_paths
                        self.person_metadata[person_id]['last_updated'] = datetime.now().isoformat()
                        self.person_metadata[person_id]['total_images'] = len(image_paths)
                        updated_count += 1
                        print(f"Updated embeddings for {person_id}: {len(all_person_embeddings)} total embeddings")
        
        if updated_count > 0 or new_persons > 0 or new_images_processed > 0:
            self.save_database()
            print(f"Database updated: {updated_count} persons updated, {new_persons} new persons, {new_images_processed} new images processed")
        else:
            print("No updates needed")
    
    def search_similar_faces(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Search for similar faces using FAISS
        
        Args:
            query_embedding: Query face embedding
            k: Number of similar faces to return
            threshold: Similarity threshold
            
        Returns:
            List of (person_id, similarity_score) tuples
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_normalized, min(k, self.faiss_index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx != -1 and similarity >= threshold:  # -1 means no result
                person_id = self.id_to_person.get(idx, "Unknown")
                results.append((person_id, float(similarity)))
        
        return results
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings and corresponding person IDs"""
        all_embeddings = []
        all_labels = []
        
        for person_id, embeddings in self.face_vectors.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)
                all_labels.append(person_id)
        
        return np.array(all_embeddings), all_labels
    
    def get_person_info(self, person_id: str) -> Dict:
        """Get metadata for a specific person"""
        return self.person_metadata.get(person_id, {})
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        total_persons = len(self.face_vectors)
        total_embeddings = sum(len(embeddings) for embeddings in self.face_vectors.values())
        
        return {
            'total_persons': total_persons,
            'total_embeddings': total_embeddings,
            'persons': list(self.face_vectors.keys())
        }
    
    def add_person(self, person_id: str, name: str, image_paths: List[str], embedder):
        """Manually add a person to the database"""
        embeddings = []
        
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                from modules.detector.app import FaceDetector
                detector = FaceDetector()
                _, detections = detector.detect_faces(image_path)
                
                if detections:
                    face_embeddings = embedder.get_embeddings([
                        cv2.resize(img[det['bbox'][1]:det['bbox'][3], det['bbox'][0]:det['bbox'][2]], (112, 112))
                        for det in detections
                    ])
                    
                    best_face_idx = np.argmax([det.get('score', 0) for det in detections])
                    embedding = face_embeddings[best_face_idx]
                    
                    # Store individual image embedding
                    self.image_embeddings[image_path] = embedding
                    embeddings.append(embedding)
                    
                    # Update image hash
                    self.image_hashes[image_path] = self.calculate_image_hash(image_path)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if embeddings:
            self.face_vectors[person_id] = np.array(embeddings)
            self.person_metadata[person_id] = {
                'name': name,
                'image_paths': image_paths,
                'last_updated': datetime.now().isoformat(),
                'total_images': len(image_paths)
            }
            self.save_database()
            return True
        
        return False
    
    def remove_person(self, person_id: str):
        """Remove a person from the database"""
        if person_id in self.face_vectors:
            # Remove individual image embeddings for this person
            if person_id in self.person_metadata:
                for image_path in self.person_metadata[person_id]['image_paths']:
                    if image_path in self.image_embeddings:
                        del self.image_embeddings[image_path]
                    if image_path in self.image_hashes:
                        del self.image_hashes[image_path]
            
            # Remove from FAISS index
            self._remove_embeddings_from_faiss(person_id)
            
            del self.face_vectors[person_id]
            del self.person_metadata[person_id]
            self.save_database()
            return True
        return False