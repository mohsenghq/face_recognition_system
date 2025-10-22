import os
import time
import cv2
import torch
import numpy as np
from insightface.model_zoo import get_model
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MILVUS_HOST = os.environ.get("MILVUS_HOST", "vector_db")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", 19530))
FACE_DB_PATH = "/app/face_db"
COLLECTION_NAME = "faces"
EMBEDDING_DIM = 512
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "buffalo_l")

# --- Face Embedder ---
class FaceEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (112, 112))
        embedding = self.model.get_embedding(img)
        return embedding

# --- Milvus Database ---
class MilvusDB:
    def __init__(self):
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        if utility.has_collection(COLLECTION_NAME):
            return Collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="person_id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields, "Face embeddings")
        return Collection(COLLECTION_NAME, schema)

    def upsert_face(self, person_id, embedding):
        self.collection.upsert([[person_id], [embedding]])
        self.collection.flush()
        print(f"Upserted {person_id} into Milvus.")

# --- Main ---
def main():
    face_embedder = FaceEmbedder()
    milvus_db = MilvusDB()
    enrolled_persons = set()

    while True:
        for person_name in os.listdir(FACE_DB_PATH):
            person_dir = os.path.join(FACE_DB_PATH, person_name)
            if os.path.isdir(person_dir) and person_name not in enrolled_persons:
                embeddings = []
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    embedding = face_embedder.get_embedding(image_path)
                    if embedding is not None:
                        embeddings.append(embedding)

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    milvus_db.upsert_face(person_name, avg_embedding)
                    enrolled_persons.add(person_name)

        time.sleep(10)

if __name__ == "__main__":
    main()
