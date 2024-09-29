import io
import json

from PIL import Image

from app.models.request_model import Coordinates
from app.services.homographic_service import HomographicService

with open("/home/kaorikizuna/Dot Vision/dot-vision/app/coordinates/example_coordinates.json", "r") as f:
    example_data = json.load(f)

validated_coordinates_data = Coordinates(**example_data)

service = HomographicService(coordinates=validated_coordinates_data)


def test_detect_person():
    image = Image.open("people_detection_test.jpg")
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="JPEG")
    image_bytes = bytes_io.getvalue()
    bytes_io.close()

    assert len(service.detect_person(frame_bytes=image_bytes)) == 2
