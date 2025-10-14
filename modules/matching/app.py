import pika
import time
import os
import json
from pymilvus import Collection, utility, connections

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "embeddings_queue"
OUTPUT_QUEUE = "matches_queue"
MILVUS_HOST = "vector_db"
MILVUS_PORT = "19530"
COLLECTION_NAME = "faces"
TOP_K = 1
SEARCH_PARAMS = {"metric_type": "L2", "params": {"nprobe": 10}}
SIMILARITY_THRESHOLD = 0.6

# --- Face Matcher Class ---
class FaceMatcher:
    """Handles connection to Milvus and face matching logic."""
    def __init__(self):
        self.collection = None
        try:
            print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            if not utility.has_collection(COLLECTION_NAME):
                # In a real system, you might have a separate service to create this.
                print(f"Warning: Milvus collection '{COLLECTION_NAME}' does not exist.")
                # For this example, we'll stop, but a real service might wait or retry.
                raise ConnectionAbortedError(f"Collection {COLLECTION_NAME} not found")

            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            print(f"Milvus collection '{COLLECTION_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Error initializing FaceMatcher: {e}")
            # The worker will keep retrying to connect in the main loop.

    def match_face(self, embedding):
        """Searches for a matching face in the Milvus collection."""
        if self.collection is None:
            # Attempt to reconnect if the collection object is missing
            self.__init__()
            if self.collection is None:
                 print("Error: Milvus connection failed, cannot match face.")
                 return "Unknown", 0.0

        try:
            results = self.collection.search(
                data=[embedding], anns_field="embedding", param=SEARCH_PARAMS,
                limit=TOP_K, output_fields=["person_id"])

            if not results or not results[0]:
                return "Unknown", 0.0

            best_match = results[0][0]
            confidence = 1.0 - best_match.distance

            if confidence > SIMILARITY_THRESHOLD:
                return best_match.entity.get("person_id"), confidence
            else:
                return "Unknown", confidence
        except Exception as e:
            print(f"Error during face matching: {e}")
            return "Unknown", 0.0

# --- RabbitMQ Worker ---
def main():
    face_matcher = FaceMatcher()

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
            embedding = message["embedding"]

            print(f"Received embedding for face from frame {frame_id} from '{INPUT_QUEUE}'")

            person_id, confidence = face_matcher.match_face(embedding)

            output_message = message.copy()
            output_message["match"] = {"person_id": person_id, "confidence": confidence}

            channel.basic_publish(
                exchange='',
                routing_key=OUTPUT_QUEUE,
                body=json.dumps(output_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            print(f"Published match '{person_id}' for face from frame {frame_id} to '{OUTPUT_QUEUE}'")

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
