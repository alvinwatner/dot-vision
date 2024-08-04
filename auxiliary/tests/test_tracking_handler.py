import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from auxiliary.tracking_handler import TrackingHandler, Frame  # Replace `your_module` with the actual module name


@pytest.fixture
def sample_frame():
    # Create a sample frame with random data
    width, height = 640, 480
    frame_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Frame(frame=frame_data, width=width, height=height)


@pytest.fixture
def sample_boxes():
    # Sample bounding boxes in TensorFlow Lite format (ymin, xmin, ymax, xmax)
    return [
        (0.1, 0.2, 0.4, 0.5),
        (0.5, 0.5, 0.8, 0.8)
    ]


@pytest.fixture
def tracking_handler():
    return TrackingHandler()


@patch('cv2.TrackerMIL_create')
def test_initialize_trackers(mock_tracker_create, tracking_handler, sample_boxes, sample_frame):
    # Mock the TrackerMIL object and its methods
    mock_tracker = MagicMock()
    mock_tracker_create.return_value = mock_tracker

    tracking_handler.initialize(sample_frame, sample_boxes)

    # Verify that the trackers are initialized
    assert len(tracking_handler.trackers) == len(sample_boxes)
    for tracker in tracking_handler.trackers:
        assert isinstance(tracker.tracker, MagicMock)
        mock_tracker.init.assert_called()


@patch('cv2.TrackerMIL_create')
def test_update_trackers(mock_tracker_create, tracking_handler, sample_boxes, sample_frame):
    # Mock the TrackerMIL object and its methods
    mock_tracker = MagicMock()
    mock_tracker_create.return_value = mock_tracker
    mock_tracker.update.return_value = (True, (10, 10, 50, 50))

    # Initialize trackers first
    tracking_handler.initialize(sample_frame, sample_boxes)

    # Create a new frame for updating
    new_frame_data = np.random.randint(0, 256, (sample_frame.height, sample_frame.width, 3), dtype=np.uint8)
    new_frame = Frame(frame=new_frame_data, width=sample_frame.width, height=sample_frame.height)

    tracking_handler.update(new_frame)

    for tracker in tracking_handler.trackers:
        assert tracker.top_left == (10, 10)
        assert tracker.bottom_right == (60, 60)


if __name__ == '__main__':
    pytest.main()
