import cv2
from auxiliary.image_handler import ImageHandler
from auxiliary.homographic_handler import HomographicHandler
from auxiliary.model_interpreter import ModelInterpreter
from auxiliary.frame_dataclass import Frame
import numpy as np


class CV2Handler:
    out: cv2.VideoWriter
    image2d_to_draw: np.ndarray[np.array([])]
    image3d_to_draw: np.ndarray[np.array([])]
    t1: float
    combined_image: np.ndarray[np.array([])]

    def __init__(self, cap: cv2.VideoCapture, image_handler: ImageHandler, homographic_handler: HomographicHandler,
                 model_interpreter: ModelInterpreter, is_save: bool, output_file: str = "output_result.mp4"):
        # image2d, image3d
        self.image_handler = image_handler

        # transform
        self.homographic_handler = homographic_handler

        # access detect and track
        self.model_interpreter = model_interpreter

        self.cap = cap
        self.is_save = is_save
        self.window_name = "Dot Vision"
        self.output_file = output_file

        self.frequency = cv2.getTickFrequency()
        self.frame_dataclass = Frame()

        if self.is_save:
            self.init_video_writer()

    def calculate_framerate(self, t1, t2):
        time = (t2 - t1) / self.frequency
        return int(1 // time)

    def _setup_cv2_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.image_handler.total_width, self.image_handler.max_height)

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 24
        self.out = cv2.VideoWriter(self.output_file, fourcc, fps,
                                   (self.image_handler.total_width, self.image_handler.max_height))

    def read_capture(self):
        success, frame = self.cap.read()
        if not success:
            return False
        self.frame_dataclass.frame = frame
        self.frame_dataclass.frame_count += 1
        return True

    def combine_image2d_image3d(self):
        self.combined_image = np.zeros((self.image_handler.max_height, self.image_handler.total_width, 3),
                                       dtype=np.uint8)

        left_height = self.image_handler.image3d.height
        left_width = self.image_handler.image3d.width

        self.combined_image[:left_height, :left_width] = self.image3d_to_draw

        # Place the 2D image on the right side of the combined image
        right_height = self.image_handler.image2d.height

        self.combined_image[:right_height, left_width:self.image_handler.total_width] = self.image2d_to_draw

    def create_bounding_box_homographic(self):
        for tracker in self.model_interpreter.tracking_handler.trackers:
            transformed_coor = self.homographic_handler.transform_coordinates(tracker.bottom_center)

            # image, position 1 (x, y), position 2 (x, y), color, thickness
            cv2.rectangle(self.image3d_to_draw, tracker.top_left, tracker.bottom_right, (255, 0, 0), 2)

            # image, center position (x, y), radius, color, thickness.
            # thickness -1 means it's a filled circle
            cv2.circle(self.image3d_to_draw, tracker.bottom_center, 4, (255, 255, 0), -1)

            # draw homographic mapping result
            # image, center position (x, y), radius, color, and thickness
            cv2.circle(self.image2d_to_draw, (int(transformed_coor[0]), int(transformed_coor[1])), 5,
                       (0, 255, 255), -2)

    def draw_framerate(self):
        t2 = cv2.getTickCount()

        frame_rate = self.calculate_framerate(self.t1, t2)

        # if below 24 fps, use red color
        if frame_rate < 24:
            color = (0, 0, 255)

        # else use light blue
        else:
            color = (255, 255, 0)

        # image, text, position (x, y), font, font scale, color, thickness
        cv2.putText(self.image3d_to_draw, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def process_frame(self, is_stream: bool = False):
        while self.read_capture():
            self.t1 = cv2.getTickCount()

            # copy the image2d and image3d to not overlap with the previous result
            self.image2d_to_draw = self.image_handler.image2d.image.copy()
            self.image3d_to_draw = cv2.resize(self.frame_dataclass.frame,
                                              (self.image_handler.image3d.width, self.image_handler.image3d.height))

            # detect and track objects
            self.model_interpreter.detect_and_track_objects(self.frame_dataclass)

            if is_stream:
                self.encode_image_for_streaming()
            else:
                self.process_and_display_frame()

        # when done, perform cleanup
        self._perform_cleanup()

    def process_and_display_frame(self):
        # create bounding box and homographic transformation
        self.create_bounding_box_homographic()

        # drawing frame rate onto image3d to draw
        self.draw_framerate()

        # combining all components together side by side
        self.combine_image2d_image3d()

        if self.is_save:
            self.out.write(self.combined_image)

        cv2.imshow(self.window_name, self.combined_image)
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:  # if user enter q or ESC key
            self._perform_cleanup()

    def _perform_cleanup(self):
        self.cap.release()
        if self.is_save:
            self.out.release()
        cv2.destroyAllWindows()
        exit()

    def encode_image_for_streaming(self):
        # extension, image
        ret, buffer = cv2.imencode(".jpg", self.combined_image)
        buffer_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame + b'\r\n')
