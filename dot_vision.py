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
parser.add_argument("--threshold", help="Set the threshold for object tracking accuracy", default=0.4)
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
    def __init__(self, model_path):
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
        self.tf_boxes = self.get_tensor_by_index(0)
        self.classes = self.get_tensor_by_index(1)
        self.scores = self.get_tensor_by_index(2)

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
        self.initialize_tracker(frame)
        ret, bbox = self.tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            return p1, p2


mod = ModelInterpreter(MODEL_PATH)


def calculate_framerate(t1, t2):
    frequency = cv2.getTickFrequency()
    time = (t2 - t1) / frequency
    framerate = 1 / time
    print(framerate)
    return round(framerate)


# first, do object detection to discover region of interest
ret, frame = cap.read()
if ret:
    mod.detect_objects(frame)

# loop through all frames in video
while True:
    # get t1 for framerate calculation
    t1 = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        print("End of the video")
        break

    # for subsequent tracking, invoke the .track_object() method
    p1, p2 = mod.track_objects(frame)
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # get t2 for framerate calculation
    t2 = cv2.getTickCount()

    # print framerate into frame
    cv2.putText(frame, f"FPS: {calculate_framerate(t1, t2)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("Dot Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:  # if user enter q or ESC key
        break

cap.release()
cv2.destroyAllWindows()
