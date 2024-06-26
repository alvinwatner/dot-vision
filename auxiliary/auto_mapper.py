import pickle
import cv2
import numpy as np
import cv2
from auxiliary.model_interpreter import ModelInterpreter
from auxiliary.tracker import Tracker
from auxiliary.utils import tuples_to_nparray


class AutoMapper:
    def __init__(self,
                 model_path,
                 threshold,
                 accelerator,
                 labels,
                 image2Ddir,
                 image3Ddir,
                 cap,
                 coors3Ddir,
                 coors2Ddir,
                 ):
        self.detector_mod = ModelInterpreter(model_path=model_path, threshold=threshold, accelerator=accelerator,
                                             labels=labels)
        self.image2d = cv2.imread(image2Ddir)
        self.image3d = cv2.imread(image3Ddir)
        self.cap = cap
        self.tracker = Tracker()
        coors2D, coors3D = self.loadCoordinates(coors2Ddir, coors3Ddir)       
        self.H, _ = cv2.findHomography(tuples_to_nparray(coors3D), tuples_to_nparray(coors2D))
        self.frame3d_height, self.frame3d_width = self.image3d.shape[:2]
        self.frame2d_height, self.frame2d_width = self.image2d.shape[:2]
        self.max_height = max(self.frame3d_height, self.frame2d_height)
        self.total_width = self.frame3d_width + self.frame2d_width


    @staticmethod
    def loadCoordinates(coors2Ddir, coors3Ddir):
        coor2d_file = open(coors2Ddir, 'rb')
        coor3d_file = open(coors3Ddir, 'rb')  
        coors2d = pickle.load(coor2d_file)
        coors3d = pickle.load(coor3d_file)                       
        return coors2d, coors3d


    def call_as_generator(self):
        """
        Display the result of ._process_frame() inside a generator method in order to generate
        a stream of frames that can be shown through a browser using a lightweight web framework, such as Flask.

        :return: frame stream separated with carriage return and new line
        """
        frame_count = 0
        while True:
            combined_image, frame_count = self._process_frame(frame_count)
            ret, buffer = cv2.imencode('.jpg', combined_image)
            if not ret:
                break
            buffer_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame + b'\r\n')

    @staticmethod
    def calculate_framerate(t1, t2):
        frequency = cv2.getTickFrequency()
        time = (t2 - t1) / frequency
        framerate = 1 / time
        return round(framerate)

    def detect_frame_at_interval(self, interval, frame, frame_count):
        """
        Detecting object and subsequently tracking them, object detection is performed
        at every frame interval.

        :param interval: frame interval to perform object detection
        :param frame: opencv2 frame object
        :param frame_count: current frame count of the video
        :return: currently tracked boxes
        """
        # every 24 frames, perform object detection and re-initialize the trackers
        if frame_count % interval == 0:
            boxes = self.detector_mod.detect_objects(frame)
            self.tracker.initialize(frame, boxes)

        # for subsequent tracking, invoke the .track_object() method
        tracked_boxes = self.tracker.update(frame)
        return tracked_boxes

    def _process_frame(self, frame_count):
        """
        Process video frames and return the combined image.

        :param frame_count: frame interval to perform object detection
        :return: combined image, frame count
        """
        # Reset image2d to the original state at the start of each iteration
        image2d = self.image2d.copy()

        # get t1 for framerate calculation
        t1 = cv2.getTickCount()

        ret, frame = self.cap.read()
        if not ret:
            print("End of the video")
            # this ._process_frame is expected to return 2 values
            return None, None

        # Resize the frame to match the dimensions of the reference image
        resized_frame = cv2.resize(frame, (self.frame3d_width, self.frame3d_height))

        tracked_boxes = self.detect_frame_at_interval(24, resized_frame, frame_count)

        # Processing each tracked box
        for (p1, p2) in tracked_boxes:
            cv2.rectangle(resized_frame, p1, p2, (255, 0, 0), 2, 1)  # Draw bounding box

            # Calculate bottom center and draw circle
            bottom_center = ((p1[0] + p2[0]) // 2, p2[1])
            cv2.circle(resized_frame, bottom_center, 4, (255, 255, 0), -1)

            source_coor = np.array([[bottom_center]], dtype='float32')
            transformed_coor = cv2.perspectiveTransform(source_coor, self.H)

            transformed_coor = np.squeeze(transformed_coor)

            cv2.circle(image2d, (int(transformed_coor[0]), int(transformed_coor[1])), radius=5, color=(0, 255, 0),
                       thickness=-2)

        # get t2 for framerate calculation
        t2 = cv2.getTickCount()

        # print framerate into frame
        frame_rate = self.calculate_framerate(t1, t2)

        # if framerate is lower than 24, display a red FPS text
        if frame_rate < 24:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.putText(resized_frame, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)

        # Create a blank image to accommodate both the frame and the PNG image
        combined_image = np.zeros((self.max_height, self.total_width, 3), dtype=np.uint8)

        # Place the video frame in the combined image
        combined_image[:self.frame3d_height, :self.frame3d_width] = resized_frame

        # Place the PNG image in the combined image next to the video frame
        combined_image[:self.frame2d_height, self.frame3d_width:self.frame3d_width + self.frame2d_width] = image2d

        frame_count += 1

        return combined_image, frame_count

    # override this method to test the app
    def final_method(self):
        """
        Final callback is designed to display whatever evaluation result it may have.
        In the original implementation, it does not have any functionality, instead it is meant
        to be overridden by its child classes.

        :return: None
        """
        pass

    def __call__(self,
                 is_stream: bool = False,
                 imshow: bool = False,
                 save_output: bool = True,
                 ):

        try:
            if is_stream:
                return self.call_as_generator()

            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_file = "output_tracker.mp4"
                fps = 24
                out = cv2.VideoWriter(output_file, fourcc, fps, (self.total_width, self.max_height))

            if imshow:
                window_name = "Dot Vision"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, self.total_width, self.max_height)

                # to perform object detection every X frame
            frame_count = 0

            # loop through all frames in video
            while True:
                combined_image, frame_count = self._process_frame(frame_count)

                if combined_image is None:
                    break

                if save_output:
                    out.write(combined_image)

                if imshow:
                    cv2.imshow(window_name, combined_image)
                    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:  # if user enter q or ESC key
                        break

            if imshow:
                self.cap.release()
                out.release()
                cv2.destroyAllWindows()


        except Exception as e:
            print(f"Error in __call__: {e}")

        self.final_method()
