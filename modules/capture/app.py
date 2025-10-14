import cv2
import pika
import time
import os
import uuid
import json
import base64

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0") # Default to "0" for webcam
FRAMES_QUEUE = "frames_queue"
FPS = int(os.environ.get("FPS", 10))
SLEEP_DURATION = 1 / FPS if FPS > 0 else 0

def main():
    """Connects to RabbitMQ, captures frames, and publishes them."""

    connection = None
    while True:
        try:
            print(f"Connecting to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300))
            print("Successfully connected to RabbitMQ.")
            break
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    channel = connection.channel()
    channel.queue_declare(queue=FRAMES_QUEUE, durable=True)

    print(f"Opening video source: {VIDEO_SOURCE}")

    def open_capture():
        try:
            # Check if VIDEO_SOURCE is a digit (for webcam index)
            video_source_int = int(VIDEO_SOURCE)
            return cv2.VideoCapture(video_source_int)
        except ValueError:
            return cv2.VideoCapture(VIDEO_SOURCE)

    cap = open_capture()

    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
        connection.close()
        return

    print("Starting frame capture...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame. Restarting capture...")
                cap.release()
                time.sleep(5)
                cap = open_capture()
                continue

            frame_id = str(uuid.uuid4())
            height, width, _ = frame.shape

            # Encode frame as JPEG for efficiency and then to base64
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            image_data_base64 = base64.b64encode(buffer).decode('utf-8')

            message = {
                "frame_id": frame_id,
                "image_data_b64": image_data_base64,
                "height": height,
                "width": width
            }

            channel.basic_publish(
                exchange='',
                routing_key=FRAMES_QUEUE,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                ))

            print(f"Published frame {frame_id} to '{FRAMES_QUEUE}'")

            if SLEEP_DURATION > 0:
                time.sleep(SLEEP_DURATION)

    except KeyboardInterrupt:
        print("Capture stopped by user.")
    finally:
        if cap.isOpened():
            cap.release()
        if connection.is_open:
            connection.close()
        print("Video capture released and RabbitMQ connection closed.")

if __name__ == '__main__':
    main()
