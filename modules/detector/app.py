import cv2
import pika
import time
import os
import json
import base64
import numpy as np
import torch
from retinaface import RetinaFace

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "frames_queue"
OUTPUT_QUEUE = "detections_queue"

# --- Face Detector Class ---
class FaceDetector:
    """Wraps the RetinaFace model for face detection."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.detector = RetinaFace(quality='normal')
        print(f"FaceDetector initialized on device: {self.device}")

    def detect_faces(self, image_bytes):
        """Detects faces in an image provided as bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Warning: Could not decode image.")
            return [], None

        faces = self.detector.predict(img)

        detections = []
        if isinstance(faces, list):
            for face in faces:
                detections.append({
                    'bbox': (face['x1'], face['y1'], face['x2'], face['y2']),
                    'confidence': face['score']
                })
        return detections, img

# --- RabbitMQ Worker ---
def main():
    face_detector = FaceDetector()

    connection = None
    while True:
        try:
            print(f"Connecting to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300))
            break
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE, durable=True)
    channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        """Processes a message from the input queue."""
        try:
            message = json.loads(body)
            frame_id = message["frame_id"]
            image_data_b64 = message["image_data_b64"]

            print(f"Received frame {frame_id} from '{INPUT_QUEUE}'")

            image_bytes = base64.b64decode(image_data_b64)
            detections, img = face_detector.detect_faces(image_bytes)

            if not detections:
                print(f"No faces found in frame {frame_id}")
            else:
                print(f"Found {len(detections)} faces in frame {frame_id}. Publishing to '{OUTPUT_QUEUE}'...")
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)

                    # Crop the detected face
                    face_crop = img[y1:y2, x1:x2]

                    # Encode face crop to base64
                    _, buffer = cv2.imencode('.png', face_crop)
                    face_crop_b64 = base64.b64encode(buffer).decode('utf-8')

                    detection_message = {
                        "frame_id": frame_id,
                        "original_image_b64": image_data_b64,
                        "face_crop_b64": face_crop_b64,
                        "bounding_box": {
                            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2
                        },
                        "confidence": det['confidence'],
                        "original_height": message["height"],
                        "original_width": message["width"]
                    }

                    channel.basic_publish(
                        exchange='',
                        routing_key=OUTPUT_QUEUE,
                        body=json.dumps(detection_message),
                        properties=pika.BasicProperties(delivery_mode=2)
                    )

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)

    try:
        print(f"Waiting for messages on '{INPUT_QUEUE}'. To exit press CTRL+C")
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Stopping consumer.")
        channel.stop_consuming()
    finally:
        connection.close()
        print("RabbitMQ connection closed.")

if __name__ == '__main__':
    main()
