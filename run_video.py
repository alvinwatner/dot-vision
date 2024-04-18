import cv2

video_source = 'samples/input_video.mp4'
cap = cv2.VideoCapture(video_source)

# loop through all frames in video
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of the video")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break    

    cv2.imshow('Dot Vision', frame)

cap.release()
cv2.destroyAllWindows()
