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


    def stream_as_image(self):
        """
        Display the result of ._process_frame() inside a generator method in order to generate
        a stream of frames that can be shown through a browser using a lightweight web framework, such as Flask.

        :return: frame stream separated with carriage return and new line
        """
        frame_count = 0
        while True:
            image, frame_count, _ = self._process_frame(frame_count, is_draw= True)
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                break
            buffer_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame + b'\r\n')
            
    def stream_raw_outputs(self):
        """
        Display the result of ._process_frame() inside a generator method in order to generate
        a stream of frames that can be shown through a browser using a lightweight web framework, such as Flask.

        :return: frame stream separated with carriage return and new line
        """
        frame_count = 0
        while True:
            _, frame_count, transformed_points = self._process_frame(frame_count)    
            yield transformed_points

    @staticmethod
    def calculate_framerate(t1, t2):
        frequency = cv2.getTickFrequency()
        time = (t2 - t1) / frequency
        framerate = 1 / time
        return round(framerate)
    
    def transform_coordinates(self, coordinates):
        """
        Perform homographic transformation on a single coordinate.

        :param coordinates: Tuple of coordinates (x, y).
        :return: Transformed coordinates as a numpy array.
        """
        source_coor = np.array([[coordinates]], dtype='float32')
        return cv2.perspectiveTransform(source_coor, self.H)    
    
    def calculate_bottom_center(p1, p2):
        """
        Calculate the bottom center point of a bounding box.

        :param p1: Tuple (x1, y1), top-left corner of the bounding box.
        :param p2: Tuple (x2, y2), bottom-right corner of the bounding box.
        :return: Tuple representing the bottom center point of the bounding box.
        """
        return (p1[0] + p2[0]) // 2, p2[1]    

    def detect_and_track_objects(self, frame, frame_count):
        """
        Detecting object and subsequently tracking them, object detection is performed
        at every frame interval.

        :param interval: frame interval to perform object detection
        :param frame: opencv2 frame object
        :param frame_count: current frame count of the video
        :return: currently tracked boxes
        """
        # every 24 frames, perform object detection and re-initialize the trackers
        if frame_count % 24 == 0:
            boxes = self.detector_mod.detect_objects(frame)
            self.tracker.initialize(frame, boxes)

        # for subsequent tracking, invoke the .track_object() method
        return self.tracker.update(frame)
    
    def draw_frame_rate(self, t1, frame3d):
        # get t2 for framerate calculation
        t2 = cv2.getTickCount()

        # print framerate into frame
        frame_rate = self.calculate_framerate(t1, t2)                

        # if framerate is lower than 24, display a red FPS text
        if frame_rate < 24:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.putText(frame3d, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)        


    def draw_detection_and_mapping(self, frame3d, frame2d, p1, p2, bottom_center, transformed_point):
        # Draw bounding box
        cv2.rectangle(frame3d, p1, p2, (255, 0, 0), 2)
        # Draw bottom center
        cv2.circle(frame3d, bottom_center, 4, (255, 255, 0), -1)
        trans_pt = transformed_point[0][0]  # Simplify the access to coordinates
        # Draw homographic mapping result
        cv2.circle(frame2d, (int(trans_pt[0]), int(trans_pt[1])), 5, (0, 255, 0), -2)        
        
    
    def transform_and_draw(self, frame2d, frame3d, boxes, is_draw):
        """
        Perform homographic transformations on detected boxes and optionally draw on the frame.

        :param frame: The frame to process.
        :param boxes: Detected bounding boxes.
        :param draw: Boolean indicating whether to draw bounding boxes and transformations.
        :return: A list of transformed points.
        """
        # get t1 for framerate calculation
        t1 = cv2.getTickCount()

        transformed_points = []
        if is_draw:
            for p1, p2 in boxes:
                bottom_center = self.calculate_bottom_center(p1, p2)
                transformed_point = self.transform_coordinates(bottom_center)
                transformed_points.append(transformed_point)
                self.draw_detection_and_mapping(frame3d, frame2d, p1, p2, bottom_center, transformed_point)

            self.draw_frame_rate(t1, frame3d)

            # Create a combined image to accommodate both the frame and the PNG image
            combined_image = np.zeros((self.max_height, self.total_width, 3), dtype=np.uint8)
            combined_image[:self.frame3d_height, :self.frame3d_width] = frame3d
            combined_image[:self.frame2d_height, self.frame3d_width:self.total_width] = frame2d

            return transformed_points, combined_image

        else:
            # If not drawing, just perform transformations
            transformed_points = [self.transform_coordinates(self.calculate_bottom_center(p1, p2)) for p1, p2 in boxes]
            return transformed_points, None        


    def _process_frame(self, frame_count, is_draw):
        """
        Process video frames and return the combined image.

        :param frame_count: frame interval to perform object detection
        :return: combined image, frame count
        """
        # Reset image2d to the original state at the start of each iteration
        image2d = self.image2d.copy()

        ret, frame = self.cap.read()
        if not ret:
            print("End of the video")
            # this ._process_frame is expected to return 3 values
            return None, None, None

        # Resize the frame to match the dimensions of the reference image
        image3d = cv2.resize(frame, (self.frame3d_width, self.frame3d_height))

        tracked_boxes = self.detect_and_track_objects(image3d, frame_count)
        transformed_points, image = self.transform_and_draw(image2d, image3d, tracked_boxes, is_draw)

        frame_count += 1

        return image, frame_count, transformed_points
    
    def stream_using_cv2(self, save_output: bool = False):
        window_name = "Dot Vision"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.total_width, self.max_height)        
        # to perform object detection every X frame
        frame_count = 0    
        # loop through all frames in video

        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = "output_tracker.mp4"
            fps = 24
            out = cv2.VideoWriter(output_file, fourcc, fps, (self.total_width, self.max_height))        

        while True:
            image, frame_count, _ = self._process_frame(frame_count, is_draw=True)            
            if image is None:
                break       
            if save_output:
                out.write(image)    
            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:  # if user enter q or ESC key
                break  
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()                                       

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
                 is_stream_as_image: bool = False,
                 is_stream_using_cv2: bool = False,
                 save_output: bool = True
                 ):

        try:
            if (not is_stream_as_image and not is_stream_using_cv2):
                return self.stream_raw_outputs()

            if is_stream_as_image:
                return self.stream_as_image()

            if is_stream_using_cv2:
                self.stream_using_cv2(save_output)
            
        except Exception as e:
            print(f"Error in __call__: {e}")

        self.final_method()
