import asyncio
import pika
import threading
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
import os
from typing import List
import time

# --- Configuration ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
INPUT_QUEUE = "tracked_results_queue"

# --- FastAPI App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        # Create a copy of the list to handle cases where a client disconnects during broadcast
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)
            except Exception as e:
                print(f"Error sending message to client: {e}")
                self.disconnect(connection)


manager = ConnectionManager()

# --- RabbitMQ Consumer in a Background Thread ---
def rabbitmq_consumer():
    connection = None
    while True:
        try:
            print(f"Consumer thread connecting to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300))
            print("Consumer thread connected to RabbitMQ.")
            break
        except pika.exceptions.AMQPConnectionError:
            print("Consumer thread failed to connect. Retrying in 5 seconds...")
            time.sleep(5)

    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE, durable=True)

    def callback(ch, method, properties, body):
        try:
            # Broadcast the raw message body to all connected WebSocket clients
            asyncio.run(manager.broadcast(body.decode('utf-8')))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error in RabbitMQ callback: {e}")

    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)

    try:
        print(f"Consumer thread waiting for messages on '{INPUT_QUEUE}'.")
        channel.start_consuming()
    except Exception as e:
        print(f"Consumer thread error: {e}")
    finally:
        if connection.is_open:
            connection.close()
        print("Consumer thread RabbitMQ connection closed.")


@app.on_event("startup")
async def startup_event():
    # Start the RabbitMQ consumer in a daemon thread
    consumer_thread = threading.Thread(target=rabbitmq_consumer, daemon=True)
    consumer_thread.start()

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
