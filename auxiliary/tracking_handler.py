import cv2
from dataclasses import dataclass
from typing import List
import logging
from auxiliary.frame_dataclass import Frame

logging.basicConfig(level=logging.DEBUG)


@dataclass
class Tracker:
    tracker: cv2.TrackerMIL

    # previously named p1, p2
    top_left: (int, int) = (0, 0)
    bottom_right: (int, int) = (0, 0)

    # computed property
    @property
    def bottom_center(self) -> (int, int):
        return (self.top_left[0] + self.bottom_right[0]) // 2, self.bottom_right[1]


class TrackingHandler:
    trackers: List[Tracker]

    def __init__(self):
        """
        initialize the tracker object

        :param frame: opencv2 frame object
        :param boxes: tensorflow lite bounding boxes
        :return: None
        """
        # the algorithm used for individual tracker
        self.tracker = cv2.TrackerMIL

        self.boxes = []

        # container for all available trackers
        self.trackers = []

    def convert_tf_boxes_to_opencv(self, boxes):
        """
        convert all tensorflow lite bounding boxes into
        opencv2 bounding box format

        :param boxes: tensorflow lite bounding boxes
        :return: None
        """
        self.boxes = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * self.frame_width)
            xmax = int(xmax * self.frame_width)
            ymin = int(ymin * self.frame_height)
            ymax = int(ymax * self.frame_height)
            width = xmax - xmin
            height = ymax - ymin

            cv_box = (xmin, ymin, width, height)
            self.boxes.append(cv_box)

    def initialize(self, frame, boxes: list):
        """
        initialize trackers for object tracking

        :param frame: opencv2 frame object
        :param boxes: tensorflow lite bounding boxes
        """

        # TODO: encapsulate frame into an object
        self.frame_height, self.frame_width, _ = frame.shape

        self.convert_tf_boxes_to_opencv(boxes)

        # remove all existing tracker
        self.trackers.clear()

        for box in self.boxes:
            tracker = self.tracker.create()
            if self._is_detection_valid(frame, box):
                tracker.init(frame, box)
            else:
                continue
            self.trackers.append(Tracker(tracker))

    @staticmethod
    def _is_detection_valid(frame, box):
        if frame is None or not frame.size:
            print("Frame is invalid")
            return False

        # bounding box is containing these values (x, y, width, height)
        if any(dim < 0 for dim in box):
            logging.debug("Bounding box dimension is invalid")
            return False
        elif box[2] <= 0 or box[3] <= 0:
            logging.debug(f"Bounding box height: {box[2]} or width: {box[3]} is invalid")
            return False
        elif box[0] + box[2] > frame.shape[1]:
            logging.debug(
                f"Bounding box x: {box[0]} and height: {box[2]} is invalid, allowed value is no more than {frame.shape[1]}")
            return False
        elif box[1] + box[3] > frame.shape[0]:
            logging.debug(
                f"Bounding box y: {box[1]} and width: {box[3]} is invalid, allowed value is no more than {frame.shape[0]}")
            return False
        return True

    def update(self, frame):
        """
        update trackers for the given frame

        :param frame: opencv2 frame object
        :return: tuple containing the bounding box for the tracked object
        """

        # TODO: get values from individual tracker
        # TODO: will break because it doesn't return anything yet
        p1_p2 = []

        # if there is no tracker, return empty list
        if not len(self.trackers):
            return p1_p2

        for index, obj in enumerate(self.trackers):
            ret, box = obj.tracker.update(frame)

            # if tracking successful, add bounding box
            if ret:
                self.trackers[index].top_left = (int(box[0]), int(box[1]))
                self.trackers[index].bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))

                # legacy reasons
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                p1_p2.append((p1, p2))

            # else, remove the tracker from memory
            else:
                self.trackers.pop(index)
        return p1_p2
