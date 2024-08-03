import argparse
from auxiliary.new_auto_mapper import AutoMapper
from web_display.display import app
from auxiliary import utils

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

model_path, labels = utils.get_model_and_labels(args)

cap = utils.get_video_capture(args)

ensemble_model = AutoMapper(
    model_path=model_path,
    threshold=args.threshold,
    accelerator=args.accelerator,
    labels=labels,
    image2Ddir=args.layout2Ddir,
    image3Ddir=args.layout3Ddir,
    coors2Ddir=args.coor2Ddir,
    coors3Ddir=args.coor3Ddir,
    cap=cap,
)

# needed for threading to work
# def run_flask_app():
#     app.run(host='0.0.0.0', port=3000, debug=False)
#

# def run_ensemble_model():
#     ensemble_model.stream_using_cv2()
#
#
# def debug_tracking_with_cv2():
#     process_web = threading.Thread(target=run_flask_app)
#     process_cv2 = threading.Thread(target=run_ensemble_model)
#     process_web.start()
#     process_cv2.start()
#     process_web.join()
#     process_cv2.join()
#
#
# debug = True

# if __name__ == '__main__':
#     if args.display == 'web':
#         if debug:
#             debug_tracking_with_cv2()
#         else:
#             run_flask_app()
#
#     if args.display == 'cv2':
#         ensemble_model.stream_using_cv2()

if __name__ == '__main__':
    if args.display == "cv2":
        ensemble_model.stream_using_cv2()
    elif args.display == "web":
        ensemble_model.stream_as_image()
    