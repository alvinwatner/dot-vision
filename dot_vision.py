import argparse
from pathlib import Path
from typing import List
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

"""
idea: using an object detection model with object tracking algorithm
model: high inference edgetpu model
tracker: lightweight object tracker
"""

# check whether to take a video from source or from live feed.
parser = argparse.ArgumentParser()
parser.add_argument("--vidsource", help="Video source for tracking")
parser.add_argument("--live", help="Enable live tracking", action="store_true")
parser.add_argument("--modeldir", help="Directory containing the detect.tflite and labelmap.txt", default="models/")
parser.add_argument("--threshold", help="Set the threshold for object tracking accuracy", default=0.7)
args = parser.parse_args()
is_live = args.live
video_source = args.vidsource
modeldir = args.modeldir
threshold = args.threshold

# set the model
MODELDIR = Path(modeldir)
MODEL_PATH = (MODELDIR / "detect.tflite").as_posix()
LABEL_PATH = (MODELDIR / "labelmap.txt").as_posix()
LABELS: List[str]

with open(LABEL_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# set capturing method
if is_live:
    cap = cv2.VideoCapture(0)
else:
    if video_source is None:
        raise ValueError("Must enter --vidsource if not using live feed")
    cap = cv2.VideoCapture(video_source)


# encapsulate interpreter to easily access its properties
class ModelInterpreter:
    def __init__(self, model_path, threshold=threshold):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]
        self.boxes: List
        self.classes: List
        self.scores: List
        self.frame_width: int
        self.frame_height: int
        self.treshold = threshold

        # all detected objects will be put in the index
        self.input_index = self.input_details[0]["index"]

        self.frame: List = []

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
        self.frame_height, self.frame_width, _ = frame.shape
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
            if scores[i] >= self.treshold:
                if LABELS[int(classes[i])] == "person":
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

    def convert_tf_boxes_to_opencv(self, tf_box):
        ymin, xmin, ymax, xmax = tf_box
        xmin = int(xmin * self.frame_width)
        xmax = int(xmax * self.frame_width)
        ymin = int(ymin * self.frame_height)
        ymax = int(ymax * self.frame_height)
        width = xmax - xmin
        height = ymax - ymin

        cv_box = (xmin, ymin, width, height)
        return cv_box

    def initialize_tracker(self, frame):
        # initializing Tracker
        self.tracker = cv2.TrackerMIL().create()
        cv_box = self.convert_tf_boxes_to_opencv(self.tf_boxes[0])
        self.tracker.init(frame, cv_box)

    def track_objects(self, frame):
        ret, bbox = self.tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            return p1, p2


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
            tracker.init(frame, box)
            self.trackers.append(tracker)

    def update(self, frame):
        """
        update trackers for the given frame

        :param frame: opencv2 frame object
        :return: tuple containing the bounding box for the tracked object
        """
        p1_p2 = []

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


mod = ModelInterpreter(MODEL_PATH)
tracker = Tracker()


def calculate_framerate(t1, t2):
    frequency = cv2.getTickFrequency()
    time = (t2 - t1) / frequency
    framerate = 1 / time
    return round(framerate)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = "output_tracker.mp4"
fps = 24
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape

out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# to perform object detection every X frame
frame_count = 0

# loop through all frames in video
while True:
    # get t1 for framerate calculation
    t1 = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        print("End of the video")
        break

    if frame_count % 24 == 0:
        boxes = mod.detect_objects(frame)
        tracker.initialize(frame, boxes)

    # for subsequent tracking, invoke the .track_object() method
    tracked_boxes = tracker.update(frame)

    for (p1, p2) in tracked_boxes:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # get t2 for framerate calculation
    t2 = cv2.getTickCount()

    # print framerate into frame
    frame_rate = calculate_framerate(t1, t2)

    # if framerate is lower than 24, display a red FPS text
    if frame_rate < 24:
        color = (0, 0, 255)
    else:
        color = (255, 255, 0)

    cv2.putText(frame, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                cv2.LINE_AA)

    out.write(frame)

    # cv2.imshow("Dot Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:  # if user enter q or ESC key
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
