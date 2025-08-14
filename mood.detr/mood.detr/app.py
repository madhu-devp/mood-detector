import cv2
import av
import time
import queue
import streamlit as st
from fer import FER
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Live Mood Tracker", page_icon="😊", layout="centered")

# ---------- CUSTOM STYLES ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    color: black !important;
}
.stApp { background: transparent; }
video { border-radius: 20px !important; }
h1, h2, h3, h4, h5, h6 {
    color: black !important; /* Titles stay black */
}
</style>
""", unsafe_allow_html=True)

# ---------- EMOJI MAP ----------
EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# ---------- VIDEO PROCESSOR ----------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.last_label_text = "Waiting for face..."

    def _analyze(self, img):
        emotions = self.detector.detect_emotions(img)
        if emotions:
            top_emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
            return top_emotion, score
        return None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = self._analyze(img)
        if result:
            label, confidence = result
            emoji = EMOJI_MAP.get(label.lower(), "🙂")
            self.last_label_text = f"{emoji} {label.capitalize()} ({confidence*100:.1f}%)"
            st.session_state.last_emotion = f"{emoji} {label.capitalize()} ({confidence*100:.1f}%)"
        else:
            self.last_label_text = "😐 No face detected"

        cv2.putText(img, self.last_label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- WEBRTC CONFIG ----------
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ---------- STREAMLIT UI ----------
# ---------- STREAMLIT UI ----------
st.markdown("<h1 style='text-align: center;'>😊 Live Mood Tracker</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Center video
with col2:
    # Initialize camera state
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    # Show start/stop buttons centered
    start_col, stop_col = st.columns([1, 1])
    with start_col:
        if st.button("▶️ Start Camera", use_container_width=True):
            st.session_state.camera_on = True
    with stop_col:
        if st.button("⏹ Stop Camera", use_container_width=True):
            st.session_state.camera_on = False

    # If camera is ON, show video feed
    if st.session_state.camera_on:
        ctx = webrtc_streamer(
            key="mood",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"width": 480, "height": 360}, "audio": False},
            video_processor_factory=EmotionProcessor,
        )

        