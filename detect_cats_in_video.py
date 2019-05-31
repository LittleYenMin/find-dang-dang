import os
import numpy as np
import cv2

from config_r_cnn import model


# Filter a list of Mask R-CNN detection results to get only the detected
# cats / dogs
def get_pet_boxes(boxes, class_ids):
    pet_boxes = []
    for box, class_id in zip(boxes, class_ids):
        if class_id in [16, 17]:
            pet_boxes.append(box)
    return np.array(pet_boxes)


def detect_objects(frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]
    return model.detect([rgb_image], verbose=0)


def draw_boxes(boxes, frame):
    for box in boxes:
        y1, x1, y2, x2 = box
        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame


# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(os.getenv('SOURCE_VIDEO'))

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Pass single image to Mask R-CNN detect pipline.
    results = detect_objects(frame)

    # Get first result cause we only pass single image to detection pipline.
    r = results[0]
    pet_boxes = get_pet_boxes(r['rois'], r['class_ids'])
    # Draw each box on the frame and show.
    cv2.imshow('Video', draw_boxes(boxes=pet_boxes, frame=frame))

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
