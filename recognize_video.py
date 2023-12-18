# Import necessary libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

# Load the pre-trained face detector model
print("Initializing Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the pre-trained face embedding model
print("Initializing Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Load the pre-trained face recognition model and label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# Start the video stream and allow the camera to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS (Frames Per Second) estimator
fps = FPS().start()

# Constants for confidence threshold and minimum face size
CONFIDENCE_THRESHOLD = 0.5
MIN_FACE_SIZE = 20

# Font settings for displaying text on the frame
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 2

# Main loop for processing video frames
while True:
    # Capture a frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create a blob from the resized image for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Use the face detector to find faces in the frame
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop through the detected faces for recognition
    i = 0
    while i < detections.shape[2]:
        confidence = detections[0, 0, i, 2]

        # Check if the confidence level meets the threshold
        if confidence > CONFIDENCE_THRESHOLD:
            # Calculate the bounding box coordinates of the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            # Extract the face region of interest (ROI)
            face = frame[startY:endY, startX:endX]
        
            # Ensure the face dimensions are sufficiently large
            if face.shape[0] >= MIN_FACE_SIZE and face.shape[1] >= MIN_FACE_SIZE:
                # Prepare the face for recognition using the embedding model
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
        
                # Perform face recognition using the trained model
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                # Display the recognition results on the frame
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)


        i += 1

    # Update the FPS counter
    fps.update()

    # Display the frame with face recognition information
    cv2.imshow("Video Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Check for the 'q' key to exit the loop
    if key == ord("q"):
        break

# Stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approximate FPS: {:.2f}".format(fps.fps()))

# Cleanup resources
cv2.destroyAllWindows()
vs.stop()
