import cv2
from dataclasses import dataclass
import numpy as np


# using data class to store image3d

@dataclass
class Image:
    # opencv object
    image: np.ndarray = np.array([])
    height: int = 0
    width: int = 0


class ImageHandler:
    image2d = Image()
    image3d = Image()
    max_height: int
    total_width: int

    def __init__(self, image2Ddir, image3Ddir):
        self.load_images(image2Ddir=image2Ddir, image3Ddir=image3Ddir)
        self.get_image_height_width()
        self.get_image_max_total()

    def load_images(self, image2Ddir, image3Ddir):
        self.image2d.image = cv2.imread(image2Ddir)
        self.image3d.image = cv2.imread(image3Ddir)

    def get_image_height_width(self):
        self.image2d.height, self.image2d.width = self.image2d.image.shape[:2]
        self.image3d.height, self.image3d.width = self.image3d.image.shape[:2]

    def get_image_max_total(self):
        self.max_height = max(self.image2d.height, self.image3d.height)
        self.total_width = self.image2d.width + self.image3d.width
