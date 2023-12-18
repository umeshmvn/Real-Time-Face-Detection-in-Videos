# Import necessary libraries
import numpy as np
import imutils
import pickle
import cv2
import os

# Function to perform face recognition on a detected face
def recognize_face(face, recognizer, le):
    # Construct a blob for the face region and pass it through the face embedding model
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                     (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    # Perform face recognition and get prediction probabilities
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]

    return name, proba

# Function to process detections and draw bounding boxes
def process_detections(detections, w, h, image, recognizer, le):
    i = 0

    while i < detections.shape[2]:
        # Extract the confidence associated with the face detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face width and height are sufficiently large
            if fW >= 20 and fH >= 20:
                # Recognize the face and get name and probability
                name, proba = recognize_face(face, recognizer, le)

                # Draw the bounding box around the face with the associated recognition probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Move to the next detection
        i += 1

# Load the pre-trained face detector model
print("Loading Face Detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the pre-trained face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# Load the pre-trained face recognition model along with the label encoder
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

# Load the input image, resize it to have a width of 600 pixels, and grab the image dimensions
image = cv2.imread('inputimg.jpg')
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# Construct a blob from the image for face detection
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

# Apply OpenCV's deep learning-based face detector to localize faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# Process detections and draw bounding boxes
process_detections(detections, w, h, image, recognizer, le)

# Save the output image
cv2.imwrite('myoutput.jpg', image)
cv2.waitKey(0)
