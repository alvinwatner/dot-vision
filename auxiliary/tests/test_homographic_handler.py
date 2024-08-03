import pytest
import numpy as np
import cv2
import pickle
from auxiliary.homographic_handler import HomographicHandler


@pytest.fixture
def mock_coordinates(tmp_path):
    coors2d = [(0, 0), (1, 0), (1, 1), (0, 1)]
    coors3d = [(10, 10), (20, 10), (20, 20), (10, 20)]
    coors2Ddir = tmp_path / "coors2d.pkl"
    coors3Ddir = tmp_path / "coors3d.pkl"
    with open(coors2Ddir, 'wb') as f:
        pickle.dump(coors2d, f)
    with open(coors3Ddir, 'wb') as f:
        pickle.dump(coors3d, f)
    return coors2Ddir, coors3Ddir, coors2d, coors3d


def test_load_coordinates(mock_coordinates):
    coors2Ddir, coors3Ddir, coors2d, coors3d = mock_coordinates
    loaded_coors2d, loaded_coors3d = HomographicHandler.load_coordinates(coors2Ddir, coors3Ddir)
    assert loaded_coors2d == coors2d
    assert loaded_coors3d == coors3d


def test_homography_initialization(mock_coordinates):
    coors2Ddir, coors3Ddir, coors2d, coors3d = mock_coordinates
    handler = HomographicHandler(coors2Ddir, coors3Ddir)
    assert handler.H is not None


def test_transform_coordinates(mock_coordinates):
    coors2Ddir, coors3Ddir, coors2d, coors3d = mock_coordinates
    handler = HomographicHandler(coors2Ddir, coors3Ddir)
    transformed_point = handler.transform_coordinates((10, 10))
    expected_transformed_point = cv2.perspectiveTransform(np.array([[[10, 10]]], dtype="float32"), handler.H)
    np.testing.assert_almost_equal(transformed_point, expected_transformed_point)
