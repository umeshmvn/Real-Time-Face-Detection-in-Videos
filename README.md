# 🎥 Real-Time Face Detection in Videos 👁️

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) 
![Python Version](https://img.shields.io/badge/Python-3.5-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-brightgreen.svg)

## 👁️ Overview

This project leverages the power of **OpenCV** and **Dlib** to perform **real-time face detection** in videos, with the added capability of recognizing faces using deep learning embeddings. The pipeline generates **128-dimensional embeddings** for detected faces and uses them to match against pre-trained facial models, making it perfect for face recognition, clustering, and identity matching.

The project implements a **face detection system** that works on both static images and live video streams. We utilize a **Caffe-based face detector** and a **deep learning model** to extract embeddings, which are later classified using an **SVM (Support Vector Machine)** for accurate real-time predictions. 

> 🚀 With **Real-Time Face Detection**, your applications can now detect and recognize faces in a fraction of a second, making it ideal for security systems, surveillance, and smart cameras.

---

## 🎯 Key Features

- 📸 **Real-time Face Detection**: Detects faces in both images and live video feeds using a pre-trained Caffe model.
- 🔬 **Deep Learning Embeddings**: Uses a **128-dimensional vector** representation for each face to recognize identities.
- 🧠 **SVM Classifier**: Classifies embeddings with an **SVM model** for face recognition.
- 🚦 **Seamless Integration**: The system integrates easily with any video or image processing pipeline for real-time applications.

---

## 📂 Project Structure

| File                    | Description                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `extract_embeddings.py`  | Script to extract facial embeddings from the dataset.                                            |
| `train_model.py`         | Script to train the SVM classifier on the extracted embeddings.                                  |
| `recognize_image.py`     | Script to test the model on static images.                                                       |
| `recognize_video.py`     | Script to detect and recognize faces in video streams.                                           |
| `res10_300x300_ssd_iter_140000.caffemodel` | Pre-trained Caffe face detector model.                                         |

---

## 🔧 Prerequisites

Ensure you have the following dependencies installed before running the scripts:

- 🐍 **Python 3.5** or higher
- 📚 **OpenCV** for real-time image and video processing
- 🔢 **NumPy** for numerical operations
- 📦 **imutils** for image processing
- 🧠 **scikit-learn** for building the SVM classifier
- 🔥 **Pre-trained Caffe Model** for face detection
- 🏋️ **Torch-based model** for generating embeddings

### Installation
You can install the necessary dependencies using:

```bash
pip install numpy opencv-python imutils scikit-learn


## 📂 Project Structure

1. **Extract Facial Embeddings**
    ```bash
    python extract_embeddings.py
    ```
    This script extracts 128D facial embeddings from the input dataset.

2. **Train the SVM Classifier**
    ```bash
    python train_model.py
    ```
    This script trains an SVM model on the extracted embeddings.

3. **Recognize Faces in Images**
    ```bash
    python recognize_image.py --image path/to/image.jpg
    ```
    Use this script to recognize faces in a static image. Replace `path/to/image.jpg` with the path to your test image.

4. **Recognize Faces in Video Streams**
    ```bash
    python recognize_video.py
    ```
    This script recognizes faces in real-time from a video stream or pre-recorded video.

## 📊 Dataset

The project uses pre-trained models to generate face embeddings, but you can also train the system on a custom dataset. Ensure that your dataset contains labeled images of different individuals and is pre-processed before running `extract_embeddings.py`.

## 🧠 How It Works

1. **Face Detection**: The system detects faces using a Caffe-based SSD model.
2. **Facial Embeddings**: Each detected face is converted into a 128-dimensional embedding using a pre-trained neural network.
3. **Face Recognition**: The SVM classifier is trained on the embeddings to predict the identity of the detected face.
4. **Video Stream Integration**: The system supports real-time face detection and recognition in live or pre-recorded video streams.

## 🛠️ Contributing

We welcome contributions! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b my-new-feature
    ```
3. Commit your changes:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin my-new-feature
    ```
5. Submit a pull request.

---

Feel free to fix bugs, improve documentation, or add new features. We appreciate all contributions!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


