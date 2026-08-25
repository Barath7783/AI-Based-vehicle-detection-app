import streamlit as st
import cv2
import tempfile
import threading

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection & Counting")


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# COCO vehicle class IDs
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]


# ---------------------------------------------------------
# VEHICLE DETECTION FUNCTION
# ---------------------------------------------------------

def detect_vehicles(frame):

    results = model(frame, verbose=False)

    vehicle_count = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes and conf > 0.4:
            vehicle_count += 1

    output_frame = results[0].plot()

    cv2.putText(
        output_frame,
        f"Vehicles: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    return output_frame, vehicle_count


# ---------------------------------------------------------
# WEBRTC VIDEO CALLBACK
# ---------------------------------------------------------

lock = threading.Lock()


def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    output_frame, vehicle_count = detect_vehicles(img)

    return av.VideoFrame.from_ndarray(
        output_frame,
        format="bgr24"
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

option = st.sidebar.radio(
    "Select Input",
    ["Webcam", "Upload Video"]
)


# =========================================================
# WEBCAM
# =========================================================

if option == "Webcam":

    st.subheader("📷 Live Webcam")

    st.info(
        "Click START below and allow camera permission in your browser."
    )

    webrtc_streamer(
        key="vehicle-detection",

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        async_processing=True
    )

    st.markdown("---")

    st.caption(
        "Live vehicle detection using YOLOv8 and WebRTC"
    )


# =========================================================
# VIDEO UPLOAD
# =========================================================

else:

    st.subheader("🎥 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        count_placeholder = st.empty()

        total_frames = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            output_frame, vehicle_count = detect_vehicles(frame)

            total_frames += 1

            frame_placeholder.image(
                output_frame,
                channels="BGR",
                use_container_width=True
            )

            count_placeholder.metric(
                "Vehicles in Current Frame",
                vehicle_count
            )

        cap.release()

        st.success("Video processing completed.")
