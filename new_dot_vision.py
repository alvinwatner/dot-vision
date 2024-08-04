from auxiliary.new_auto_mapper import AutoMapper
from auxiliary import utils
from auxiliary.dot_vision_arguments import dot_vision_arguments

"""
idea: using an object detection model with object tracking algorithm
model: high inference edgetpu/cpu model
tracker: lightweight object tracker
"""

args = dot_vision_arguments()

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

if __name__ == '__main__':
    if args.display == "cv2":
        ensemble_model.stream_using_cv2()
    elif args.display == "web":
        from web_display.backend import app
        for rule in app.url_map.iter_rules():
            print(rule)
        app.run()
