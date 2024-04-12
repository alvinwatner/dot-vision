import cv2
from pathlib import Path

image_dir = Path("../ground_truths/")

images = sorted([file for file in image_dir.iterdir() if file.is_file()])

# iterate through the images, one image at a time, press "q" to exit
for image in images:
    frame = cv2.imread(image.as_posix())
    cv2.imshow("Dot Vision", frame)
    key_pressed = cv2.waitKey(0)
    if key_pressed & 0xFF == ord("q"):
        break
