from app.config import logger
from ultralytics import YOLO
from abc import ABC
from typing import Any, Optional, Union
from celery import Celery, Task
from app.config import CONFIDENCE_THRESHOLD
from app.utils.image_helpers import get_image_from_bytes


class BaseService(ABC):
    def __init__(self, celery_app: Optional[Celery] = None):
        self.model = YOLO("yolov8n.pt")
        self.logger = logger

        self.celery_app = celery_app

    def execute(self, *args: Any, **kwargs):
        """
        Task executor for dot vision. Dynamically adapt to either use a Celery message broker
        or asynchronous operation from fastapi.

        Will automatically execute ONE method with prefix "perform_"
        """
        # Find the method with the name that contains 'perform_'
        task_func_name = next(
            (method_name for method_name in dir(self) if method_name.startswith("perform_")),
            None
        )

        if not task_func_name:
            raise ValueError("No method found with 'perform_' in the name")

        # Get the actual method
        task_func = getattr(self, task_func_name)

        if self.celery_app:
            return task_func.delay(*args, **kwargs)
        else:
            return task_func(*args, **kwargs)

    def detect_person(self, frame_bytes):
        # read from image, convert to RGB, and get the numpy array
        img = get_image_from_bytes(frame_bytes)

        results = self.model(img)

        boxes = results[0].boxes
        class_names = results[0].names

        detected_people = []

        for box in boxes:
            cls_name = class_names[int(box.cls)]
            if cls_name == "person":
                if box.conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bottom_center = self.calculate_bottom_center(x1, y1, x2, y2)
                    detected_people.append(bottom_center)

        return detected_people

    @staticmethod
    def calculate_bottom_center(x1, y1, x2, y2):
        return (x1 + x2) // 2, y1
