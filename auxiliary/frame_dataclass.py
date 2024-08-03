from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    # cv2 frame object
    frame: np.ndarray = np.array([])
    width: int = 0
    height: int = 0
    frame_count: int = 0
