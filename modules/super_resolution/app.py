import cv2
import pika
import time
import os
import json
import base64
import numpy as np
import torch
from gfpgan import GFPGANer

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "detections_queue"
OUTPUT_QUEUE = "upscaled_faces_queue"
MODEL_PATH = '/app/weights/GFPGANv1.4.pth'

# --- Face Enhancer Class ---
class FaceEnhancer:
    """Wraps the GFPGAN model for face super-resolution."""
    def __init__(self, model_path=MODEL_PATH, upscale=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.enhancer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
        print(f"FaceEnhancer initialized on device: {device}")

    def enhance_face(self, image_bytes):
        """Enhances a single face image provided as bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_crop is None:
            print("Warning: Could not decode face crop for enhancement.")
            return None

        _, _, restored_face = self.enhancer.enhance(
            img_crop, has_aligned=False, only_center_face=True, paste_back=False)

        if restored_face is None:
            print("Warning: Face enhancement failed.")
            return None

        _, buffer = cv2.imencode('.png', restored_face)
        return buffer.tobytes()

# --- RabbitMQ Worker ---
def main():
    face_enhancer = FaceEnhancer()

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
            face_crop_b64 = message["face_crop_b64"]

            print(f"Received face from frame {frame_id} from '{INPUT_QUEUE}'")

            face_crop_bytes = base64.b64decode(face_crop_b64)
            upscaled_face_bytes = face_enhancer.enhance_face(face_crop_bytes)

            if upscaled_face_bytes:
                upscaled_face_b64 = base64.b64encode(upscaled_face_bytes).decode('utf-8')

                output_message = message.copy()
                output_message["upscaled_face_b64"] = upscaled_face_b64

                channel.basic_publish(
                    exchange='',
                    routing_key=OUTPUT_QUEUE,
                    body=json.dumps(output_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(f"Published upscaled face from frame {frame_id} to '{OUTPUT_QUEUE}'")
            else:
                print(f"Skipping face from frame {frame_id} due to enhancement failure.")

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
