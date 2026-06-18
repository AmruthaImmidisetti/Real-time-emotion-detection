# Real-Time Emotion Detection System

## Overview

A real-time facial emotion recognition application built using Python,
OpenCV, DeepFace, and Streamlit. The system captures live webcam video,
detects faces, analyzes facial expressions, and predicts emotions such
as Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.

------------------------------------------------------------------------

## Features

-   Real-time webcam emotion detection
-   Face detection using OpenCV Haar Cascade Classifier
-   Emotion recognition using DeepFace pre-trained models
-   Browser-based interface using Streamlit
-   Live emotion labels with confidence scores
-   Real-time video streaming with Streamlit-WebRTC

------------------------------------------------------------------------

## Tech Stack

### Programming Language

-   Python

### Libraries & Frameworks

-   OpenCV
-   DeepFace
-   Streamlit
-   Streamlit-WebRTC
-   NumPy

### AI / ML Concepts

-   Artificial Intelligence
-   Deep Learning
-   Computer Vision
-   Facial Emotion Recognition

------------------------------------------------------------------------

## Project Workflow

``` text
Webcam
   ↓
Capture Video Frame
   ↓
Face Detection (OpenCV)
   ↓
Face Extraction
   ↓
Emotion Analysis (DeepFace)
   ↓
Emotion Prediction
   ↓
Display Results in Browser
```

------------------------------------------------------------------------

## Installation

### Clone Repository

``` bash
git clone <your-github-repository-url>
cd emotion-detection
```

### Install Dependencies

``` bash
pip install opencv-python
pip install deepface
pip install streamlit
pip install streamlit-webrtc
pip install tf-keras
```

Or

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Run the Application

``` bash
streamlit run detector.py
```

Open:

``` text
http://localhost:8501
```

Allow camera access when prompted.

------------------------------------------------------------------------

## Emotions Detected

-   Happy 😊
-   Sad 😔
-   Angry 😠
-   Fear 😨
-   Surprise 😲
-   Neutral 😐
-   Disgust 🤢

------------------------------------------------------------------------

## Challenges Faced

-   Webcam integration in Streamlit
-   Real-time video processing
-   DeepFace dependency management
-   TensorFlow and NumPy compatibility issues
-   Emotion prediction accuracy under varying lighting conditions

------------------------------------------------------------------------

## Future Enhancements

-   Emotion history tracking
-   Emotion analytics dashboard
-   Multiple face detection
-   Emotion-based recommendations
-   Cloud deployment

------------------------------------------------------------------------

## Author

**Amrutha Varshini Immidisetti**
