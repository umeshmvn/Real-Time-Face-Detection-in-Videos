# USAGE
# Command to execute the script: python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import argparse
import pickle

# Load the pre-computed face embeddings
print("Loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# Encode the labels for training
print("Encoding labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data["names"])

# Train the face recognition model using the 128-dimensional embeddings
print("Training the face recognition model...")
face_recognizer = SVC(C=1.0, kernel="linear", probability=True)
face_recognizer.fit(data["embeddings"], encoded_labels)

# Save the trained face recognition model to disk
with open("output/recognizer.pickle", "wb") as recognizer_file:
    recognizer_file.write(pickle.dumps(face_recognizer))

# Save the label encoder to disk
with open("output/le.pickle", "wb") as label_encoder_file:
    label_encoder_file.write(pickle.dumps(label_encoder))
