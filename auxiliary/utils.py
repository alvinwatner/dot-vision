import cv2
import numpy as np


def tuples_to_nparray(tuple_list):
    # Convert each tuple in the list to a list
    return np.array([list(t) for t in tuple_list], dtype='float32')


def nparray_to_tuples(np_array):
    # Convert numpy array back to a list of tuples, assuming the shape of np_array is (n, 2)
    return [tuple(map(int, np.round(row))) for row in np_array]
