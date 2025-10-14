import pika
import time
import os
import json
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "matches_queue"
OUTPUT_QUEUE = "tracked_results_queue"
DEFAULT_FPS = 30
DEFAULT_TRACK_BUFFER = 30

# --- Tracker Wrapper ---
class TrackerWrapper:
    """A wrapper for the BYTETracker algorithm."""
    def __init__(self, fps=DEFAULT_FPS, track_buffer=DEFAULT_TRACK_BUFFER):
        self.tracker = BYTETracker(fps=fps, track_buffer=track_buffer)

    def track_faces(self, match_results, image_shape):
        """Updates the tracker with detections from a single frame."""
        if not match_results:
            self.tracker.update(np.empty((0, 5)), image_shape, image_shape)
            return []

        detections = []
        for res in match_results:
            bbox = res["bounding_box"]
            score = res["match"]["confidence"] if res["match"]["person_id"] != "Unknown" else 0.1
            detections.append([bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"], score])

        online_targets = self.tracker.update(np.array(detections), image_shape, image_shape)

        tracked_results = []
        for t in online_targets:
            track_id = t.track_id
            tlbr = t.tlbr
            for res in match_results:
                bbox = res["bounding_box"]
                if np.allclose(tlbr, [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]], atol=1):
                    res["track_id"] = track_id
                    tracked_results.append(res)
                    break

        return tracked_results

# --- RabbitMQ Worker ---
class TrackerWorker:
    def __init__(self):
        self.tracker_wrapper = TrackerWrapper()
        self.frame_buffer = {}
        self.last_processed_frame_id = None
        self.connection = self._connect_to_rabbitmq()
        self.channel = self.connection.channel()
        self._setup_queues()

    def _connect_to_rabbitmq(self):
        while True:
            try:
                return pika.BlockingConnection(pika.ConnectionParameters(
                    host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300))
            except pika.exceptions.AMQPConnectionError:
                print("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
                time.sleep(5)

    def _setup_queues(self):
        self.channel.queue_declare(queue=INPUT_QUEUE, durable=True)
        self.channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
        self.channel.basic_qos(prefetch_count=100)

    def process_frame_buffer(self, frame_id_to_process):
        """Processes a completed frame from the buffer."""
        if frame_id_to_process in self.frame_buffer:
            match_results = self.frame_buffer.pop(frame_id_to_process)
            print(f"Processing frame {frame_id_to_process} with {len(match_results)} matches.")

            if not match_results:
                return

            # Get image shape from the first message in the batch
            image_shape = (match_results[0]['original_height'], match_results[0]['original_width'])

            tracked_results = self.tracker_wrapper.track_faces(match_results, image_shape)

            if tracked_results:
                original_message = match_results[0]
                output_message = {
                    "frame_id": frame_id_to_process,
                    "original_image_b64": original_message.get("original_image_b64"),
                    "original_height": image_shape[0],
                    "original_width": image_shape[1],
                    "tracked_persons": tracked_results
                }

                self.channel.basic_publish(
                    exchange='',
                    routing_key=OUTPUT_QUEUE,
                    body=json.dumps(output_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(f"Published tracking results for frame {frame_id_to_process} to '{OUTPUT_QUEUE}'")

    def callback(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            frame_id = message["frame_id"]

            if self.last_processed_frame_id and frame_id != self.last_processed_frame_id:
                self.process_frame_buffer(self.last_processed_frame_id)

            if frame_id not in self.frame_buffer:
                self.frame_buffer[frame_id] = []
            self.frame_buffer[frame_id].append(message)

            self.last_processed_frame_id = frame_id

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        self.channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=self.callback)
        try:
            print(f"Waiting for messages on '{INPUT_QUEUE}'. To exit press CTRL+C")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("Stopping consumer.")
        if self.last_processed_frame_id:
            self.process_frame_buffer(self.last_processed_frame_id)
        self.channel.stop_consuming()
        self.connection.close()
        print("RabbitMQ connection closed.")

if __name__ == '__main__':
    worker = TrackerWorker()
    worker.start()
