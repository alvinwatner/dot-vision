import argparse
from pathlib import Path

import cv2
import numpy as np


def tuples_to_nparray(tuple_list):
    # Convert each tuple in the list to a list
    return np.array([list(t) for t in tuple_list], dtype='float32')


def nparray_to_tuples(np_array):
    # Convert numpy array back to a list of tuples, assuming the shape of np_array is (n, 2)
    return [tuple(map(int, np.round(row))) for row in np_array]


def get_video_capture(args: argparse.Namespace):
    if args.live:
        cap = cv2.VideoCapture(0)
    else:
        if args.vidsource is None or args.vidsource == "":
            raise ValueError("Must enter --vidsource if not using live feed")
        cap = cv2.VideoCapture(args.vidsource)
    return cap


def get_model_and_labels(args: argparse.Namespace):
    model_dir = Path(args.modeldir)
    if args.accelerator == "cpu":
        model_path = (model_dir / "detect.tflite").as_posix()
    elif args.accelerator == "tpu":
        model_path = (model_dir / "edgetpu.tflite").as_posix()
    else:
        raise ValueError("Unknown accelerator option")
    label_path = (model_dir / "labelmap.txt").as_posix()

    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return model_path, labels
