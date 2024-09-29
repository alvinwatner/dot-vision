import cv2
import numpy as np

from app.models.request_model import Coordinates
from app.services.base_service import BaseService


class HomographicService(BaseService):
    def __init__(self, coordinates: Coordinates, celery_app=None):
        super().__init__(celery_app)
        source_coordinates, destination_coordinates = self.load_coordinates(coordinates)
        self.H, _ = cv2.findHomography(source_coordinates, destination_coordinates)

    @staticmethod
    def load_coordinates(coordinates):
        src_coors = np.array([[point.x, point.y] for point in coordinates.source_coordinates])
        dst_coors = np.array([[point.x, point.y] for point in coordinates.destination_coordinates])

        return src_coors, dst_coors

    def perform_homographic_transformation(self, frame_bytes):
        detected_people = self.detect_person(frame_bytes)
        homographed_people = []
        for bottom_center in detected_people:
            source_coor = np.array([[bottom_center]], dtype="float32")
            transformed_coor = cv2.perspectiveTransform(source_coor, self.H)[0][0]
            homographed_people.append(list(map(int, transformed_coor)))
        print(homographed_people)
        return homographed_people
