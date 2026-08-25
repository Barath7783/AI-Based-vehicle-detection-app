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

        # -------------------------------------------------
        # SHOW ORIGINAL VIDEO
        # -------------------------------------------------

        st.subheader("▶️ Original Video")

        st.video(uploaded_file)

        # -------------------------------------------------
        # SAVE UPLOADED VIDEO
        # -------------------------------------------------

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(uploaded_file.getbuffer())
        tfile.close()

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Unable to open uploaded video.")
        else:

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # -------------------------------------------------
            # OUTPUT VIDEO
            # -------------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # -------------------------------------------------
            # DISPLAY PLACEHOLDERS
            # -------------------------------------------------

            frame_placeholder = st.empty()
            count_placeholder = st.empty()

            progress = st.progress(0)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            frame_number = 0

            max_vehicle_count = 0

            # -------------------------------------------------
            # PROCESS VIDEO
            # -------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    verbose=False
                )

                count = 0

                for box in results[0].boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls in vehicle_classes and conf > 0.4:
                        count += 1

                # Draw YOLO detections
                output_frame = results[0].plot()

                # Add vehicle count
                cv2.putText(
                    output_frame,
                    f"Vehicles: {count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Write processed frame
                out.write(output_frame)

                # Show current processed frame
                frame_placeholder.image(
                    output_frame,
                    channels="BGR",
                    use_container_width=True
                )

                count_placeholder.metric(
                    "Vehicles in Current Frame",
                    count
                )

                if count > max_vehicle_count:
                    max_vehicle_count = count

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(
                            frame_number / total_frames,
                            1.0
                        )
                    )

            # -------------------------------------------------
            # RELEASE VIDEO
            # -------------------------------------------------

            cap.release()
            out.release()

            progress.progress(1.0)

            st.success(
                "✅ Video processing completed!"
            )

            # -------------------------------------------------
            # SHOW PROCESSED VIDEO
            # -------------------------------------------------

            st.subheader("🚗 Processed Video")

            with open(output_path, "rb") as video_file:

                processed_video = video_file.read()

            st.video(processed_video)

            st.metric(
                "Maximum Vehicles in a Frame",
                max_vehicle_count
            )
