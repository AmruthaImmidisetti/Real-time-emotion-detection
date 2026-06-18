import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from deepface import DeepFace

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Real-Time Emotion Detector")
st.write("Live emotion detection using DeepFace + Streamlit")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            try:
                padding = 20

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.shape[1], x + w + padding)
                y2 = min(img.shape[0], y + h + padding)

                face = img[y1:y2, x1:x2]

                result = DeepFace.analyze(
                    face,
                    actions=["emotion"],
                    enforce_detection=True,
                    detector_backend="opencv"
                )

                emotion = result[0]["dominant_emotion"]
                score = result[0]["emotion"][emotion]
                all_scores = result[0]["emotion"]
                # print(all_scores)

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"{emotion} ({score:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    webrtc_streamer(
        key="emotion-detector",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )