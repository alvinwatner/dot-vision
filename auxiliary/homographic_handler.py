import cv2
from auxiliary.utils import tuples_to_nparray
import numpy as np
import pickle


class HomographicHandler:
    """
    Handle homographic transform being performed
    """

    def __init__(self, coors2Ddir, coors3Ddir):
        coors2d, coors3d = self.load_coordinates(coors2Ddir, coors3Ddir)
        self.H, _ = cv2.findHomography(tuples_to_nparray(coors3d), tuples_to_nparray(coors2d))

    @staticmethod
    def load_coordinates(coors2Ddir, coors3Ddir):
        with open(coors2Ddir, 'rb') as coor2d_file, open(coors3Ddir, 'rb') as coor3d_file:
            coors2d = pickle.load(coor2d_file)
            coors3d = pickle.load(coor3d_file)
        return coors2d, coors3d

    def transform_coordinates(self, coordinates):
        source_coor = np.array([[coordinates]], dtype="float32")
        return cv2.perspectiveTransform(source_coor, self.H)[0][0]
