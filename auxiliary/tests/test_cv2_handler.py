import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from auxiliary.cv2_handler import CV2Handler
from auxiliary.image_handler import ImageHandler
from auxiliary.homographic_handler import HomographicHandler
from auxiliary.model_interpreter import ModelInterpreter
from auxiliary.frame_dataclass import Frame


@pytest.fixture
def mock_video_capture():
    return MagicMock(spec=cv2.VideoCapture)


@pytest.fixture
def mock_image_handler():
    handler = MagicMock(spec=ImageHandler)
    handler.total_width = 1280
    handler.max_height = 720
    handler.image2d.image = np.zeros((720, 640, 3), dtype=np.uint8)
    handler.image3d.image = np.zeros((720, 640, 3), dtype=np.uint8)
    return handler


@pytest.fixture
def mock_homographic_handler():
    return MagicMock(spec=HomographicHandler)


@pytest.fixture
def mock_model_interpreter():
    interpreter = MagicMock(spec=ModelInterpreter)
    interpreter.tracking_handler.trackers = []
    return interpreter


@pytest.fixture
def cv2_handler(mock_video_capture, mock_image_handler, mock_homographic_handler, mock_model_interpreter):
    return CV2Handler(
        cap=mock_video_capture,
        image_handler=mock_image_handler,
        homographic_handler=mock_homographic_handler,
        model_interpreter=mock_model_interpreter,
        is_save=False
    )


def test_calculate_framerate(cv2_handler):
    t1 = 0
    t2 = cv2.getTickFrequency() * 2  # 2 seconds later
    assert cv2_handler.calculate_framerate(t1, t2) == 2


def test_read_capture_success(cv2_handler, mock_video_capture):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_video_capture.read.return_value = (True, frame)

    assert cv2_handler.read_capture() is True
    assert cv2_handler.frame_dataclass.frame_count == 1
    assert np.array_equal(cv2_handler.frame_dataclass.frame_dataclass, frame)


def test_read_capture_failure(cv2_handler, mock_video_capture):
    mock_video_capture.read.return_value = (False, None)

    assert cv2_handler.read_capture() is False


def test_combine_image2d_image3d(cv2_handler, mock_image_handler):
    cv2_handler.image2d_to_draw = np.ones((720, 640, 3), dtype=np.uint8) * 255
    cv2_handler.image3d_to_draw = np.ones((720, 640, 3), dtype=np.uint8) * 127

    cv2_handler.combine_image2d_image3d()

    combined_image = cv2_handler.combined_image
    assert combined_image.shape == (720, 1280, 3)
    assert np.array_equal(combined_image[:, :640], cv2_handler.image3d_to_draw)
    assert np.array_equal(combined_image[:, 640:], cv2_handler.image2d_to_draw)


def test_draw_framerate(cv2_handler):
    cv2_handler.image3d_to_draw = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2_handler.t1 = cv2.getTickCount()
    t2 = cv2_handler.t1 + cv2.getTickFrequency() * 2  # 2 seconds later

    with patch('cv2.getTickCount', return_value=t2):
        cv2_handler.draw_framerate()

    assert "FPS: 2" in cv2_handler.image3d_to_draw.tobytes()


def test_draw_result(cv2_handler):
    cv2_handler.image2d_to_draw = np.zeros((720, 640, 3), dtype=np.uint8)
    cv2_handler.image3d_to_draw = np.zeros((720, 640, 3), dtype=np.uint8)

    cv2_handler.process_and_display_frame()

    combined_image = cv2_handler.combined_image
    assert combined_image.shape == (720, 1280, 3)


if __name__ == '__main__':
    pytest.main()
