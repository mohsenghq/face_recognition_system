import cv2
import pika
import time
import os
import json
import base64
import numpy as np
import torch
from insightface.model_zoo import get_model

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "upscaled_faces_queue"
OUTPUT_QUEUE = "embeddings_queue"

# --- Face Embedder Class ---
class FaceEmbedder:
    """Wraps the InsightFace ArcFace model for face embedding generation."""
    def __init__(self, model_name='buffalo_l', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        ctx_id = 0 if self.device == 'cuda' else -1
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=ctx_id)
        print(f"FaceEmbedder initialized on device: {self.device}")

    def get_embedding(self, image_bytes):
        """Generates an embedding for a single face image provided as bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None:
            print("Warning: Could not decode image for embedding.")
            return None

        face_resized = cv2.resize(face_img, (112, 112))
        embedding = self.model.get_embedding(face_resized)
        return embedding

# --- RabbitMQ Worker ---
def main():
    face_embedder = FaceEmbedder()

    connection = None
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300))
            break
        except pika.exceptions.AMQPConnectionError:
            print("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
            time.sleep(5)

    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE, durable=True)
    channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            message = json.loads(body)
            frame_id = message["frame_id"]
            upscaled_face_b64 = message["upscaled_face_b64"]

            print(f"Received upscaled face from frame {frame_id} from '{INPUT_QUEUE}'")

            upscaled_face_bytes = base64.b64decode(upscaled_face_b64)
            embedding = face_embedder.get_embedding(upscaled_face_bytes)

            if embedding is not None:
                output_message = message.copy()
                output_message["embedding"] = embedding.tolist()

                channel.basic_publish(
                    exchange='',
                    routing_key=OUTPUT_QUEUE,
                    body=json.dumps(output_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(f"Published embedding for face from frame {frame_id} to '{OUTPUT_QUEUE}'")
            else:
                print(f"Skipping embedding for face from frame {frame_id} due to failure.")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)

    try:
        print(f"Waiting for messages on '{INPUT_QUEUE}'. To exit press CTRL+C")
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()

if __name__ == '__main__':
    main()
