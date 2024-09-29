import base64
import binascii
import io

from PIL import Image
from pydantic import BaseModel, conlist, model_validator, ValidationError, field_validator


class FrameRequest(BaseModel):
    frame: str  # base64 string

    @field_validator("frame")
    @classmethod
    def decode_frame(cls, value: str):
        try:
            image_bytes = base64.b64decode(value)
        except (Exception, binascii.Error):
            raise ValidationError("Invalid base64 code")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.copy().verify()  # verify header
            image.copy().load()  # load pixel data into memory (prevent truncated images)
            image.convert("RGB")  # check if can be decoded with RGB scheme
        except (IOError, OSError):
            raise ValidationError("Not an image")
        return image_bytes


class Points(BaseModel):
    x: int
    y: int


class Coordinates(BaseModel):
    # set minimum coordinate length here
    source_coordinates: conlist(Points, min_length=4)
    destination_coordinates: conlist(Points, min_length=4)

    @model_validator(mode='after')
    def check_equal_length(self):
        if len(self.source_coordinates) != len(self.destination_coordinates):
            raise ValidationError('source coordinates and destination coordinates must have the same length')
        return self
