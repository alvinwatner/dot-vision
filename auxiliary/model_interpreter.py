from tflite_runtime.interpreter import Interpreter, load_delegate
from typing import List
import cv2
import numpy as np
from auxiliary.frame_dataclass import Frame
from auxiliary.tracking_handler import TrackingHandler


class ModelInterpreter:
    frame_dataclass: Frame

    def __init__(self, model_path, threshold, accelerator, labels, frame_interval=24):
        """
        Initialize model to interpreter wrapper

        Args:
            model_path (str): Path to the trained model.
            threshold (float): Detection threshold.
            accelerator (str): Hardware acceleration option.
            labels (list[str]): List of class labels.
            frame_interval (int): Interval to perform object detection (default: 24)
        """
        self.labels = labels
        if accelerator == "cpu":
            self.interpreter = Interpreter(model_path=model_path)
        elif accelerator == "tpu":
            self.interpreter = Interpreter(model_path=model_path,
                                           experimental_delegates=[load_delegate("libedgetpu.so.1.0")])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.threshold = threshold

        # all detected objects will be put in the index
        self.input_index = self.input_details[0]["index"]

        # self.frame: List = []
        self.frame_interval = frame_interval
        self.tracking_handler = TrackingHandler()

    def preprocess_image(self):
        frame_to_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_to_rgb, (self.width, self.height))

        # turn 3D array into 4D [1xHxWx3]
        expanded_dims = np.expand_dims(frame_resized, axis=0)
        self.frame = expanded_dims

    """
    input: frame
    output: boxes, classes, score

    get the frame, set the input, get the output
    """

    def detect_objects(self, frame):
        self.frame = frame
        self.preprocess_image()
        self.interpreter.set_tensor(self.input_index, self.frame)
        self.interpreter.invoke()

        # tensorflow boxes has a different bounding box format than opencv
        return self.filter_boxes()

    def filter_boxes(self):
        tf_boxes = self.get_tensor_by_index(0)
        classes = self.get_tensor_by_index(1)
        scores = self.get_tensor_by_index(2)

        filtered_boxes = []
        for i in range(len(scores)):
            if scores[i] >= self.threshold:
                if self.labels[int(classes[i])] == "person":
                    filtered_boxes.append(tf_boxes[i])
        return filtered_boxes

    def get_tensor_by_index(self, index):
        """
        indices:
        0 for boxes
        1 for classes
        2 for confidence scores
        """
        return self.interpreter.get_tensor(self.output_details[index]["index"])[0]

    def detect_and_track_objects(self, frame: Frame):
        if frame.frame_count % self.frame_interval == 0:
            boxes = self.detect_objects(frame.frame)
            self.tracking_handler.initialize(frame.frame, boxes)
        return self.tracking_handler.update(frame.frame)
