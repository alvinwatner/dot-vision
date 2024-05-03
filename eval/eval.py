import argparse
from pathlib import Path
from typing import List
import json
import cv2
from sklearn.metrics import mean_squared_error

import statistics

from auxiliary.auto_mapper import AutoMapper

with open("../models/labelmap.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture("../samples/input_video.mp4")

parser = argparse.ArgumentParser()
parser.add_argument("--coor2Ddir", help="2D coordinates data", default='../coordinates/2d_coordinates.pkl')
parser.add_argument("--coor3Ddir", help="3D coordinates data", default='../coordinates/3d_coordinates.pkl')
parser.add_argument("--layout2Ddir", help="2D layout image", default='../coordinates/2d_image.png')
parser.add_argument("--layout3Ddir", help="3D layout image", default='../coordinates/3d_image.png')
parser.add_argument("--gtdir", help="Directory containing the ground truth file", default='ground_truths.txt')
parser.add_argument("--modeldir", help="Directory containing the detect.tflite and labelmap.txt", default="../models/")
parser.add_argument("--threshold", help="Set the threshold for object tracking accuracy", default=0.6)
parser.add_argument("--write_predictions", help="Whether to write the predictions or not.", action="store_true")
parser.add_argument("--accelerator", help="Set the accelerator used in object detection", choices=["cpu", "tpu"],
                    default="cpu")
args = parser.parse_args()
modeldir = args.modeldir
threshold = args.threshold
accelerator = args.accelerator
image2Ddir = args.layout2Ddir
image3Ddir = args.layout3Ddir
coor2Ddir = args.coor2Ddir
coor3Ddir = args.coor3Ddir
ground_truth = args.gtdir
write_predictions = args.write_predictions

# set the model
MODELDIR = Path(modeldir)
if accelerator == "cpu":
    MODEL_PATH = (MODELDIR / "detect.tflite").as_posix()
elif accelerator == "tpu":
    MODEL_PATH = (MODELDIR / "edgetpu.tflite").as_posix()
LABEL_PATH = (MODELDIR / "labelmap.txt").as_posix()
LABELS: List[str]

class EvalAutoMapper(AutoMapper):
    def __init__(self, model_path, threshold, accelerator, labels, image2Ddir, image3Ddir, cap, coors3Ddir, coors2Ddir,
                 ground_truth):
        with open(ground_truth, "r") as f:
            # type casting into integer to allow comparison between integer and integer
            self.ground_truths = [int(line.strip()) for line in f.readlines()]

        self.predicted_value = []
        self.framerates = []
        self.predict_info = []
        super().__init__(model_path, threshold, accelerator, labels, image2Ddir, image3Ddir, cap, coors3Ddir, coors2Ddir)
    
    def calculate_framerate(self, t1, t2):
        framerate = super().calculate_framerate(t1, t2)
        self.framerates.append(framerate)
        return framerate

    def detect_frame_at_interval(self, interval, frame, frame_count):
        tracked_boxes = super().detect_frame_at_interval(interval, frame, frame_count)
        self.predicted_value.append(len(tracked_boxes))
        self.predict_info.append({"prediction" : len(tracked_boxes), "ground_truth" : self.ground_truths[frame_count]})
        return tracked_boxes

    def final_method(self):
        mse = mean_squared_error(y_true=self.ground_truths, y_pred=self.predicted_value)
        avg_fps = statistics.mean(self.framerates)
        stdev_fps = statistics.stdev(self.framerates)
        scores = {
            "mse": mse,
            "avg_fps": avg_fps,
            'stdev_fps': stdev_fps
        }
        # Serializing json
        scores = json.dumps(scores, indent=4)
        
        # Writing to sample.json
        with open("metric_scores.json", "w") as outfile:
            outfile.write(scores)     

        if write_predictions:
            predict_info = {'data': self.predict_info}
            predict_info_json = json.dumps(predict_info, indent=4)
            with open("predictions.json", "w") as outfile:
                outfile.write(predict_info_json)              

        print("Mean Squared Error: {}".format(mse))
        print("Average Framerate: {}".format(avg_fps))
        print("Framerate Std: {}".format(stdev_fps))
        
modeldir = args.modeldir

ensemble_model = EvalAutoMapper(
    model_path=MODEL_PATH,
    threshold=threshold,
    accelerator=accelerator,
    labels=LABELS,
    image2Ddir=image2Ddir,
    image3Ddir=image3Ddir,
    cap=cap,
    coors2Ddir=coor2Ddir,
    coors3Ddir=coor3Ddir,
    ground_truth=ground_truth
)

ensemble_model(imshow=False)