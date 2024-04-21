import argparse
from pathlib import Path
from typing import List
import cv2
import pickle
import cv2
from auxiliary.auto_mapper import AutoMapper
from web_display.display import app

"""
idea: using an object detection model with object tracking algorithm
model: high inference edgetpu model
tracker: lightweight object tracker
"""

# check whether to take a video from source or from live feed.
parser = argparse.ArgumentParser()
parser.add_argument("--display", help="Display output target", choices=["cv2", "web"], default="web")
parser.add_argument("--vidsource", help="Video source for tracking", default='samples/input_video.mp4')
parser.add_argument("--layout2Ddir", help="2D layout image", default='coordinates/2d_image.png')
parser.add_argument("--layout3Ddir", help="3D layout image", default='coordinates/3d_image.png')
parser.add_argument("--coor2Ddir", help="2D coordinates data", default='coordinates/2d_coordinates.pkl')
parser.add_argument("--coor3Ddir", help="3D coordinates data", default='coordinates/3d_coordinates.pkl')
parser.add_argument("--live", help="Enable live tracking", action="store_true")
parser.add_argument("--modeldir", help="Directory containing the detect.tflite and labelmap.txt", default="models/")
parser.add_argument("--threshold", help="Set the threshold for object tracking accuracy", default=0.6)
parser.add_argument("--accelerator", help="Set the accelerator used in object detection", choices=["cpu", "tpu"],
                    default="tpu")
args = parser.parse_args()
is_live = args.live
video_source = args.vidsource
modeldir = args.modeldir
threshold = args.threshold
image2Ddir = args.layout2Ddir
image3Ddir = args.layout3Ddir
accelerator = args.accelerator
display = args.display

coor2Ddir = args.coor2Ddir
coor3Ddir = args.coor3Ddir

coor2d_file = open(coor2Ddir, 'rb')
coor3d_file = open(coor3Ddir, 'rb')

# This coors2d and coors3d created from extract_coordinate.py tool.
coors2d = pickle.load(coor2d_file)
coors3d = pickle.load(coor3d_file)

# set the model
MODELDIR = Path(modeldir)
if accelerator == "cpu":
    MODEL_PATH = (MODELDIR / "detect.tflite").as_posix()
elif accelerator == "tpu":
    MODEL_PATH = (MODELDIR / "edgetpu.tflite").as_posix()
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

ensemble_model = AutoMapper(
    model_path = MODEL_PATH,
    threshold= threshold,
    accelerator=accelerator, 
    labels=LABELS,
    image2Ddir=image2Ddir,
    image3Ddir=image3Ddir,
    cap=cap,
    coors3d=coors3d,
    coors2d=coors2d
)
if __name__ == '__main__':
    if (display == 'web'):
        print("Display is set to web, invoking auto_mapper")
        app.run(host='0.0.0.0', port=3001)

    if (display == 'cv2'):
        print("Display is set to cv2, invoking auto_mapper")
        ensemble_model.run(imshow=True, save_output=True)
        print("auto_mapper has been called")
