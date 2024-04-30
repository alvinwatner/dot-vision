import cv2
import pickle
from sklearn.metrics import mean_squared_error
from auxiliary.auto_mapper import AutoMapper
import statistics

with open("../models/labelmap.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture("../samples/input_video.mp4")

coor2d_file = open('../coordinates/2d_coordinates.pkl', 'rb')
coor3d_file = open('../coordinates/3d_coordinates.pkl', 'rb')

# This coors2d and coors3d created from extract_coordinate.py tool.
coors2d = pickle.load(coor2d_file)
coors3d = pickle.load(coor3d_file)


class EvalAutoMapper(AutoMapper):
    def __init__(self, model_path, threshold, accelerator, labels, image2Ddir, image3Ddir, cap, coors3d, coors2d,
                 ground_truth):
        with open(ground_truth, "r") as f:
            # type casting into integer to allow comparison between integer and integer
            self.ground_truth = [int(line.strip()) for line in f.readlines()]

        self.predicted_value = []
        self.framerates = []
        super().__init__(model_path, threshold, accelerator, labels, image2Ddir, image3Ddir, cap, coors3d, coors2d)

    def evaluate_mse(self, tracked_boxes):
        self.predicted_value.append(len(tracked_boxes))

    def calculate_framerate(self, t1, t2):
        framerate = super().calculate_framerate(t1, t2)
        self.framerates.append(framerate)
        return framerate

    def detect_frame_at_interval(self, interval, frame, frame_count):
        tracked_boxes = super().detect_frame_at_interval(interval, frame, frame_count)
        self.evaluate_mse(tracked_boxes)
        return tracked_boxes

    def final_method(self):
        print(
            "Mean Squared Error: {}".format(mean_squared_error(y_true=self.ground_truth, y_pred=self.predicted_value)))
        print("Average Framerate: {}".format(statistics.mean(self.framerates)))


ensemble_model = EvalAutoMapper(
    model_path="../models/detect.tflite",
    threshold=0.6,
    accelerator="cpu",
    labels=LABELS,
    image2Ddir="../coordinates/2d_image.png",
    image3Ddir="../coordinates/3d_image.png",
    cap=cap,
    coors3d=coors3d,
    coors2d=coors2d,
    ground_truth="ground_truths.txt"
)

ensemble_model(imshow=True)
