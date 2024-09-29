import pytest
from pydantic import ValidationError

from app.models.request_model import FrameRequest


def test_frame_request_model_fail_not_image():
    with pytest.raises(ValidationError) as excinfo:
        FrameRequest(frame="aGVsbG8gd29ybGQK")

