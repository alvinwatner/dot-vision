import cv2

# Path to the video file
video_path = '/home/kaorikizuna/Dot Vision/dot-vision/samples/nick_room-6.mkv'

# Timestamp in seconds (e.g., 120 seconds)
timestamp = 7

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame number corresponding to the timestamp
frame_number = int(fps * timestamp)

# Set the video position to the calculated frame number
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame at the desired timestamp
ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
else:
    # Save or display the frame
    cv2.imwrite('frame_at_{}s.jpg'.format(timestamp), frame)
    # cv2.imshow('Frame at {} seconds'.format(timestamp), frame)
    # cv2.waitKey(0)

# When everything done, release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
