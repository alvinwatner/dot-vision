from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    # cv2 frame object
    frame: np.ndarray = np.array([])
    frame_count: int = 0

    @property
    def height(self):
        return self.frame.shape[0]

    def width(self):
        return self.frame.shape[1]
