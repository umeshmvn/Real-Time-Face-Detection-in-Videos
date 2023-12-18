## Overview of OpenFace for a single input image
1. Identify faces using pre-existing models from dlib or OpenCV libraries.
2. Adjust the detected face for neural network analysis. This project utilizes dlib’s real-time facial landmark detection combined with OpenCV's affine transformations to normalize the facial structure, aiming to align the eyes and bottom lip consistently across different images.
3. Leverage a deep neural network to generate a 128-dimensional representation of the face on a unit hypersphere. This representation, or embedding, serves as a versatile facial identifier. Distinctive to this method is the ability for the distance between any two facial embeddings to signify the likelihood of identity disparity, which simplifies tasks such as clustering, similarity detection, and classification.
4. Employ clustering or classification algorithms on these embeddings to accomplish the face recognition process.

## Project Structure
extract_embeddings.py - Script to extract facial embeddings from the dataset.
train_model.py - Script to train the SVM classifier on the extracted embeddings.
recognize_image.py - Script to test the model on a static image.
recognize_video.py - Script to test the model on video streams.

## Prerequisites
The project requires the following dependencies:

--> Python 3.5
--> OpenCV
--> NumPy
--> imutils
--> scikit-learn
--> Pre-trained Caffe-based face detector model files
--> Pre-trained Torch-based face recognition model (openface_nn4.small2.v1.t7)

## Running the Scripts
--> Extract Facial Embeddings: python extract_embeddings.py
--> Train the SVM Classifier: python train_model.py
--> Recognize Faces in Images:python recognize_image.py --image path/to/image.jpg
--> Recognize Faces in Video Streams: python recognize_video.py
