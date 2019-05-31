import os
import numpy as np
import cv2
import dotenv

dotenv.load_dotenv()


# Filter a list of Mask R-CNN detection results to get only the detected cats / dogs
def get_pet_boxes(boxes, class_ids):
    pet_boxes = []
    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [16, 17]:
            pet_boxes.append(box)
    return np.array(pet_boxes)


# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(os.getenv('SOURCE_VIDEO'))


if __name__ == '__main__':
    from config_r_cnn import model

    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        # Run the image through the Mask R-CNN model to get results.
        results = model.detect([rgb_image], verbose=0)

        # Mask R-CNN assumes we are running detection on multiple images.
        # We only passed in one image to detect, so only grab the first result.
        r = results[0]

        # The r variable will now have the results of detection:
        # - r['rois'] are the bounding box of each detected object
        # - r['class_ids'] are the class id (type) of each detected object
        # - r['scores'] are the confidence scores for each detection
        # - r['masks'] are the object masks for each detected object (which gives you the object outline)

        # Filter the results to only grab the car / truck bounding boxes
        pet_boxes = get_pet_boxes(r['rois'], r['class_ids'])

        print("Pets found in frame of video:")

        # Draw each box on the frame
        for box in pet_boxes:
            print("pet: ", box)

            y1, x1, y2, x2 = box

            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Show the frame of video on the screen
        cv2.imshow('Video', frame)

        # Hit 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up everything when finished
    video_capture.release()
    cv2.destroyAllWindows()
