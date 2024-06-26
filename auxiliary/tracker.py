import cv2


class Tracker:

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
        # self.initialize(frame, boxes)

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

    def initialize(self, frame, boxes):
        """
        initialize trackers for object tracking

        :param frame: opencv2 frame object
        :param boxes: tensorflow lite bounding boxes
        """
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
            self.trackers.append(tracker)

    @staticmethod
    def _is_detection_valid(frame, box):
        if frame is None or not frame.size:
            print("Frame is invalid")
            return False

        # bounding box is containing these values (x, y, width, height)
        if any(dim < 0 for dim in box):
            print("Bounding box dimension is invalid")
            return False
        elif box[2] <= 0 or box[3] <= 0:
            print(f"Bounding box height: {box[2]} or width: {box[3]} is invalid")
            return False
        elif box[0] + box[2] > frame.shape[1]:
            print(f"Bounding box x: {box[0]} and height: {box[2]} is invalid, allowed value is no more than {frame.shape[1]}")
            return False
        elif box[1] + box[3] > frame.shape[0]:
            print(f"Bounding box y: {box[1]} and width: {box[3]} is invalid, allowed value is no more than {frame.shape[0]}")
            return False
        return True

    def update(self, frame):
        """
        update trackers for the given frame

        :param frame: opencv2 frame object
        :return: tuple containing the bounding box for the tracked object
        """
        p1_p2 = []

        # if there is no tracker, return empty list
        if not len(self.trackers):
            return p1_p2

        for index, tracker in enumerate(self.trackers):
            ret, box = tracker.update(frame)

            # if tracking successful, add bounding box
            if ret:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                p1_p2.append((p1, p2))

            # else, remove the tracker from memory
            else:
                self.trackers.pop(index)
        return p1_p2
