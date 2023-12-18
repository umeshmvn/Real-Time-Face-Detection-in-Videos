# Import necessary libraries
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from tqdm import tqdm  # Import tqdm for progress bar


CONF_THRESHOLD = 0.5

# Load the pre-trained face detector model
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the pre-trained face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Get the file paths of the input images in the dataset
print("Quantifying Faces...")
imagePaths = list(paths.list_images("dataset"))

# Initialize lists to store facial embeddings and corresponding names
knownEmbeddings = []
knownNames = []

# Initialize a counter for the total number of faces processed
total = 0

# Loop over the image paths with tqdm for a progress bar
for (i, imagePath) in enumerate(tqdm(imagePaths, desc="Processing images")):
    # Extract the person's name from the image path
    name = imagePath.split(os.path.sep)[-2]

    # Load the image, resize it, and get its dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Construct a blob from the image for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Apply face detection to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if detections.shape[2] > 0:
        # find the index of the face with the highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the highest confidence meets the threshold
        if confidence > CONF_THRESHOLD:
            # extract the face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region and check if it meets the size requirement
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW >= 20 and fH >= 20:
                # Construct a blob for the face region and pass it through the face embedding model
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Add the name and face embedding to the respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

# dump the facial embeddings + names to disk using the 'with' statement
output_file_path = "output/embeddings.pickle"
with open(output_file_path, "wb") as f:
    f.write(pickle.dumps({"embeddings": knownEmbeddings, "names": knownNames}))

print(f"[INFO] Serializing {total} encodings to {output_file_path}")