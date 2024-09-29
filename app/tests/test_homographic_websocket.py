import io

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app


def test_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws/homograhpic") as websocket:
        image = Image.open("people_detection_test.jpg")
        bytes_io = io.BytesIO()
        image.save(bytes_io, format="JPEG")
        image_bytes = bytes_io.getvalue()
        bytes_io.close()

        websocket.send_json({"frame": image_bytes})
