import argparse
import copy
from pathlib import Path
from typing import List
import cv2
import pickle
import cv2
from auxiliary.auto_mapper import AutoMapper
from web_display.backend import app
import multiprocessing
import threading

"""
idea: using an object detection model with object tracking algorithm
model: high inference edgetpu model
tracker: lightweight object tracker
"""

# check whether to take a video from source or from live feed.
parser = argparse.ArgumentParser()
parser.add_argument("--display", help="Display output target", choices=["cv2", "web"], default="cv2")
parser.add_argument("--vidsource", help="Video source for tracking", default='samples/nick_room-6.mkv')
parser.add_argument("--layout2Ddir", help="2D layout image", default='coordinates/nick/image_2D.png')
parser.add_argument("--layout3Ddir", help="3D layout image", default='coordinates/nick/image_3D.png')
parser.add_argument("--coor2Ddir", help="2D coordinates data",
                    default='coordinates/nick/coordinates_2D.pkl')
parser.add_argument("--coor3Ddir", help="3D coordinates data",
                    default='coordinates/nick/coordinates_3D.pkl')
parser.add_argument("--live", help="Enable live tracking", action="store_true")
parser.add_argument("--modeldir", help="Directory containing the detect.tflite and labelmap.txt", default="models/")
parser.add_argument("--threshold", help="Set the threshold for object tracking accuracy", default=0.6)
parser.add_argument("--accelerator", help="Set the accelerator used in object detection", choices=["cpu", "tpu"],
                    default="cpu")
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
    model_path=MODEL_PATH,
    threshold=threshold,
    accelerator=accelerator,
    labels=LABELS,
    image2Ddir=image2Ddir,
    image3Ddir=image3Ddir,
    coors2Ddir=coor2Ddir,
    coors3Ddir=coor3Ddir,
    cap=cap,
)


# needed for threading to work
def run_flask_app():
    app.run(host='0.0.0.0', port=3000, debug=False)


def run_ensemble_model():
    ensemble_model(is_stream_using_cv2=True, save_output=True)


def debug_tracking_with_cv2():
    process_web = threading.Thread(target=run_flask_app)
    process_cv2 = threading.Thread(target=run_ensemble_model)
    process_web.start()
    process_cv2.start()
    process_web.join()
    process_cv2.join()


debug = False

if __name__ == '__main__':
    if display == 'web':
        if debug:
            debug_tracking_with_cv2()
        else:
            run_flask_app()

    if display == 'cv2':
        ensemble_model(is_stream_using_cv2=True, save_output=True)
