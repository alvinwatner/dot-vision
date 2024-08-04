import cv2

from auxiliary.cv2_handler import CV2Handler
from auxiliary.homographic_handler import HomographicHandler
from auxiliary.image_handler import ImageHandler
from auxiliary.model_interpreter import ModelInterpreter


class AutoMapper:
    def __init__(self, model_path: str, threshold: float, accelerator: str, labels: list[str], image2Ddir: str,
                 image3Ddir: str,
                 cap: cv2.VideoCapture,
                 coors3Ddir: str, coors2Ddir: str,
                 **kwargs):
        model_interpreter = ModelInterpreter(model_path, threshold, accelerator, labels,
                                             kwargs.get("frame_interval", 24))
        self.image_handler = ImageHandler(image2Ddir, image3Ddir)
        self.homographic_handler = HomographicHandler(coors2Ddir, coors3Ddir)
        self.cv2_handler = CV2Handler(cap=cap,
                                      image_handler=self.image_handler,
                                      homographic_handler=self.homographic_handler,
                                      model_interpreter=model_interpreter,
                                      is_save=kwargs.get("is_save", False),
                                      output_file=kwargs.get("output_file", "output_result.mp4"))

    def generate_raw_outputs(self):
        """
        Reading the frame, performing detection and tracking and then returning transformed coordinates
        """
        # if frame can be read, perform detection, then return
        if self.cv2_handler.read_capture():
            self.cv2_handler.model_interpreter.detect_and_track_objects(self.cv2_handler.frame_dataclass)
            return [self.homographic_handler.transform_coordinates(tracker.bottom_center) for tracker in
                    self.cv2_handler.model_interpreter.tracking_handler.trackers]

    def stream_using_cv2(self):
        self.cv2_handler.process_frame()

    def stream_as_image(self):
        self.cv2_handler.process_frame(draw_as_image=True)
