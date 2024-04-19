import cv2
import numpy as np

def calculate_framerate(t1, t2):
    frequency = cv2.getTickFrequency()
    time = (t2 - t1) / frequency
    framerate = 1 / time
    return round(framerate)


def tuples_to_nparray(tuple_list):
    # Convert each tuple in the list to a list
    return np.array([list(t) for t in tuple_list], dtype='float32')


def nparray_to_tuples(np_array):
    # Convert numpy array back to a list of tuples, assuming the shape of np_array is (n, 2)
    return [tuple(map(int, np.round(row))) for row in np_array]

def process_video_frames(original_image2d, cap, tracker, mod, frame_width, frame_height, h, max_height, total_width, out, window_name, accelerator):
    frame_count = 0
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
            # Draw bounding box
            cv2.rectangle(resized_frame, p1, p2, (255, 0, 0), 2, 1)

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

        # Calculate and display the frame rate
        frame_rate = calculate_framerate(t1, t2)
        color = (0, 0, 255) if frame_rate < 24 else (255, 255, 0)
        cv2.putText(resized_frame, f"FPS: {frame_rate}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Create a blank image to accommodate both the frame and the PNG image
        combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        combined_image[:frame_height, :frame_width] = resized_frame
        combined_image[:image_height, frame_width:frame_width + image_width] = image2d

        out.write(combined_image)

        if accelerator != "tpu":
            cv2.imshow(window_name, combined_image)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:  # if user presses 'q' or ESC key
            break

        frame_count += 1