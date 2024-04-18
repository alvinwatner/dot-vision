import argparse
from pathlib import Path
from typing import List
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate
import numpy as np
import pickle
from flask import Flask, Response, render_template_string
import cv2
from auxiliary.model_interpreter import ModelInterpreter
from auxiliary.tracker import Tracker
from auxiliary.utils import calculate_framerate, tuples_to_nparray, nparray_to_tuples
from web_display.display import app

"""
idea: using an object detection model with object tracking algorithm
model: high inference edgetpu model
tracker: lightweight object tracker
"""

# check whether to take a video from source or from live feed.
parser = argparse.ArgumentParser()
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

# encapsulate interpreter to easily access its properties
mod = ModelInterpreter(model_path=MODEL_PATH, threshold=threshold, accelerator=accelerator, labels=LABELS)
tracker = Tracker()

# convert coors from list of tuples to list of list
pts_src = tuples_to_nparray(coors3d)
pts_dst = tuples_to_nparray(coors2d)

# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = "output_tracker.mp4"
fps = 24
ret, frame = cap.read()
image2d = cv2.imread(image2Ddir)

# This is necessary as for the 3D->2D mapping to works.
# Because the homography matrix created from predefined coordinates
# that extracted from a specific image size. This image size for 3D and
# 2D need to be preserved on each frame
image3D = cv2.imread(image3Ddir)
if image3D is None:
    print(f"Failed to load 3D layout image from {image3D}")
    exit()
# Video frame dimensions
frame_height, frame_width = image3D.shape[:2]

# Ensure the video is opened successfully
if not cap.isOpened():
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Read the first frame to get the video frame size (assuming all frames are of the same size)
ret, frame = cap.read()
if not ret:
    print("Failed to read the first frame of the video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# PNG image dimensions
image_height, image_width = image2d.shape[:2]

# Calculate combined dimensions
max_height = max(frame_height, image_height)
total_width = frame_width + image_width

window_name = "Dot Vision"

# Adjust the window size for the combined image
# headless off
# assuming acceleration tpu means headless
# if accelerator != "tpu":
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, total_width, max_height)

out = cv2.VideoWriter(output_file, fourcc, fps, (total_width, max_height))

original_image2d = image2d.copy()


def gen_frames():
    # to perform object detection every X frame
    frame_count = 0

    # loop through all frames in video
    while True:
        # Reset image2d to the original state at the start of each iteration
        image2d = original_image2d.copy()

        # get t1 for framerate calculation
        t1 = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            print("End of the video")
            break

        # Resize the frame to match the dimensions of the reference image
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        if frame_count % 24 == 0:
            boxes = mod.detect_objects(resized_frame)
            tracker.initialize(resized_frame, boxes)

        # for subsequent tracking, invoke the .track_object() method
        tracked_boxes = tracker.update(resized_frame)

        # Process each tracked box for bottom center calculation, drawing on frame, and transformation
        for (p1, p2) in tracked_boxes:
            cv2.rectangle(resized_frame, p1, p2, (255, 0, 0), 2, 1)  # Draw bounding box

            # Calculate bottom center and draw circle
            bottom_center = ((p1[0] + p2[0]) // 2, p2[1])
            cv2.circle(resized_frame, bottom_center, 4, (255, 255, 0), -1)

            source_coor = np.array([[bottom_center]], dtype='float32')
            transformed_coor = cv2.perspectiveTransform(source_coor, h)

            transformed_coor = np.squeeze(transformed_coor)

            cv2.circle(image2d, (int(transformed_coor[0]), int(transformed_coor[1])), radius=5, color=(0, 255, 0),
                       thickness=-2)

        # get t2 for framerate calculation
        t2 = cv2.getTickCount()

        # print framerate into frame
        frame_rate = calculate_framerate(t1, t2)

        # if framerate is lower than 24, display a red FPS text
        if frame_rate < 24:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.putText(resized_frame, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)

        # Create a blank image to accommodate both the frame and the PNG image
        combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

        # Place the video frame in the combined image
        combined_image[:frame_height, :frame_width] = resized_frame

        # Place the PNG image in the combined image next to the video frame
        combined_image[:image_height, frame_width:frame_width + image_width] = image2d

        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        frame_count += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frame data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
