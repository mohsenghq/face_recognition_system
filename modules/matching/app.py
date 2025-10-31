# matcher.py
import numpy as np
from typing import List, Optional

class FaceMatcher:
    def __init__(self, vector_db=None, threshold=0.46):
        """
        Initialize face matcher with FAISS-based vector database
        
        Args:
            vector_db: FaceVectorDB instance with FAISS index
            threshold: similarity threshold for matching
        """
        self.vector_db = vector_db
        self.threshold = threshold

    def match(self, embeddings: np.ndarray) -> List[str]:
        """
        Match face embeddings against known faces using FAISS
        
        Args:
            embeddings: Face embeddings to match
            
        Returns:
            List of matched person IDs or "Unknown"
        """
        if self.vector_db is None:
            return ["Unknown"] * len(embeddings)
        
        results = []
        for embedding in embeddings:
            # Search for similar faces using FAISS
            similar_faces = self.vector_db.search_similar_faces(
                embedding, 
                k=1, 
                threshold=self.threshold
            )
            
            if similar_faces:
                # Return the most similar person
                person_id, similarity = similar_faces[0]
                results.append(person_id)
            else:
                results.append("Unknown")
        
        return results
    
    def set_vector_db(self, vector_db):
        """Update the vector database reference"""
        self.vector_db = vector_db
