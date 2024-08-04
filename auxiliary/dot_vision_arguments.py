import argparse


def dot_vision_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", help="Display output target", choices=["cv2", "web"], default="web")
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
    return args
