import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from collections import Counter
import time

st.set_page_config(page_title="Emotion Detector", layout="wide")
st.title("Real-Time Emotion Detector")
st.markdown("Uses **DeepFace** + **OpenCV** to detect your emotions via webcam.")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

col1, col2 = st.columns(2)
emotion_label = col1.empty()
confidence_label = col2.empty()
chart_placeholder = st.empty()

emotion_history = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = None

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=False
                )
                emotion = result[0]["dominant_emotion"]
                score   = result[0]["emotion"][emotion]

                emotion_history.append(emotion)
                emotion_label.metric("Emotion", emotion.capitalize())
                confidence_label.metric("Confidence", f"{score:.1f}%")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({score:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2
                )
            except:
                pass

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)

        if emotion_history:
            counts = Counter(emotion_history[-50:])
            chart_placeholder.bar_chart(counts)

        time.sleep(0.05)

if cap:
    cap.release()