import numpy as np
from PIL import Image
import io
from base64 import b64encode


def get_image_from_bytes(frame_bytes):
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    return np.array(img)


def get_base64_from_image(filepath: str):
    with Image.open(filepath) as img:
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="JPEG")
        print(b64encode(image_bytes.getvalue()))
