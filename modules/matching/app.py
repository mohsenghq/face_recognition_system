# matcher.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceMatcher:
    def __init__(self, known_embeddings, known_labels, threshold=0.5):
        """
        known_embeddings: np.array (N, D)
        known_labels: known people
        threshold: similarity threshold
        """
        self.known_embeddings = known_embeddings
        self.known_labels = known_labels
        self.threshold = threshold

    def match(self, embeddings):
        results = []
        for emb in embeddings:
            sims = cosine_similarity([emb], self.known_embeddings)[0]
            max_idx = np.argmax(sims)
            if sims[max_idx] > self.threshold:
                results.append(self.known_labels[max_idx])
            else:
                results.append("Unknown")
        return results
