import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from insightface.model_zoo import get_model
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- Milvus Configuration ---
MILVUS_HOST = os.environ.get("MILVUS_HOST", "vector_db")
MILVUS_PORT = "19530"
COLLECTION_NAME = "faces"
DIMENSION = 512  # Based on the ArcFace model
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"

# --- Face Embedder Class (copied from embedding service for consistency) ---
class FaceEmbedder:
    """Wraps the InsightFace ArcFace model for face embedding generation."""
    def __init__(self, model_name='arcface_r100_v1', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        ctx_id = 0 if self.device == 'cuda' else -1
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=ctx_id)
        print(f"FaceEmbedder initialized on device: {self.device}")

    def get_embedding(self, image_path):
        """Generates an embedding for a single face image from a file path."""
        face_img = cv2.imread(image_path)
        if face_img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
        face_resized = cv2.resize(face_img, (112, 112))
        embedding = self.model.get_embedding(face_resized)
        return embedding

# --- Milvus Management Functions ---
def connect_to_milvus():
    """Establishes connection to the Milvus server."""
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus.")

def create_milvus_collection():
    """Creates the 'faces' collection in Milvus if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="person_id", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, "Face recognition collection")
    collection = Collection(COLLECTION_NAME, schema)

    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    print(f"Successfully created collection '{COLLECTION_NAME}' with index.")

def clear_milvus_collection():
    """Deletes all data from the 'faces' collection."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Nothing to clear.")
        return

    collection = Collection(COLLECTION_NAME)
    # A simple delete query that matches all entities
    collection.delete('person_id != ""')
    collection.flush()
    print(f"Successfully cleared all data from collection '{COLLECTION_NAME}'.")


def enroll_faces(image_dir: str):
    """Enrolls faces from a directory into the Milvus collection."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist. Please create it first.")
        return

    collection = Collection(COLLECTION_NAME)
    embedder = FaceEmbedder()

    person_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    print(f"Found {len(person_dirs)} persons to enroll.")

    for person_id in tqdm(person_dirs, desc="Enrolling Persons"):
        person_path = os.path.join(image_dir, person_id)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        embeddings = []
        person_ids = []

        for image_name in image_files:
            image_path = os.path.join(person_path, image_name)
            embedding = embedder.get_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)
                person_ids.append(person_id)

        if embeddings:
            collection.insert([embeddings, person_ids])

    collection.flush()
    print(f"\nEnrollment complete. Total entities in collection: {collection.num_entities}")


# --- Main CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus Face Database Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-parser for 'create-collection'
    parser_create = subparsers.add_parser("create-collection", help="Create the Milvus collection.")

    # Sub-parser for 'clear-collection'
    parser_clear = subparsers.add_parser("clear-collection", help="Delete all data from the collection.")

    # Sub-parser for 'enroll'
    parser_enroll = subparsers.add_parser("enroll", help="Enroll faces from a directory.")
    parser_enroll.add_argument("--path", type=str, required=True, help="Path to the directory containing face images. Subdirectory names are used as person_id.")

    args = parser.parse_args()

    connect_to_milvus()

    if args.command == "create-collection":
        create_milvus_collection()
    elif args.command == "clear-collection":
        clear_milvus_collection()
    elif args.command == "enroll":
        enroll_faces(args.path)

    connections.disconnect("default")
    print("Milvus connection closed.")
