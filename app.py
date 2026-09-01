import os
import time
import tempfile
import threading

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Vehicle Detection & Tracking")
st.caption("AI-based vehicle detection, counting and live tracking")


# ============================================================
# VEHICLE CLASSES - COCO
# ============================================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        st.info("Downloading YOLOv8 model...")
    
    return YOLO(model_path)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

input_type = st.sidebar.radio(
    "Select Input",
    ["Upload Video", "Live Webcam"],
    index=0
)

confidence = st.sidebar.slider(
    "Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

show_labels = st.sidebar.checkbox(
    "Show Vehicle Labels",
    value=True
)

show_ids = st.sidebar.checkbox(
    "Show Tracking IDs",
    value=True
)


# ============================================================
# VEHICLE COLORS
# ============================================================

def vehicle_color(class_id):
    colors = {
        2: (0, 255, 0),       # Car
        3: (255, 0, 0),       # Motorcycle
        5: (0, 165, 255),     # Bus
        7: (255, 0, 255)      # Truck
    }

    return colors.get(class_id, (255, 255, 255))


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_tracking(frame, result):

    output = frame.copy()

    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    if result.boxes is None:
        return output, counts, 0

    boxes = result.boxes

    total = 0

    for i in range(len(boxes)):

        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        if cls not in VEHICLE_CLASSES:
            continue

        if conf < confidence:
            continue

        vehicle_name = VEHICLE_CLASSES[cls]

        counts[vehicle_name] += 1
        total += 1

        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

        x1, y1, x2, y2 = xyxy

        color = vehicle_color(cls)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ----------------------------------------------------
        # TRACKING ID
        # ----------------------------------------------------

        track_id = None

        if boxes.id is not None:
            track_id = int(boxes.id[i].item())

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = ""

        if show_labels:
            label = f"{vehicle_name} {conf:.2f}"

        if show_ids and track_id is not None:
            label += f" | ID: {track_id}"

        if label:

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 30)),
                (x1 + max(120, len(label) * 9), y1),
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (x1 + 5, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_text = (
        f"Vehicles: {total} | "
        f"Cars: {counts['Car']} | "
        f"Motorcycles: {counts['Motorcycle']} | "
        f"Buses: {counts['Bus']} | "
        f"Trucks: {counts['Truck']}"
    )

    cv2.rectangle(
        output,
        (10, 10),
        (min(output.shape[1] - 10, 850), 50),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        output,
        panel_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output, counts, total


# ============================================================
# UPLOAD VIDEO
# ============================================================

if input_type == "Upload Video":

    st.header("🎥 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a vehicle video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a road/traffic video for vehicle detection and tracking."
    )

    if uploaded_file is not None:

        # ----------------------------------------------------
        # SAVE TEMP VIDEO
        # ----------------------------------------------------

        suffix = os.path.splitext(uploaded_file.name)[1]

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_input.write(uploaded_file.read())
        temp_input.close()

        # ----------------------------------------------------
        # SHOW ORIGINAL VIDEO
        # ----------------------------------------------------

        st.subheader("🎬 Original Video")

        st.video(uploaded_file)

        # ----------------------------------------------------
        # START BUTTON
        # ----------------------------------------------------

        start_tracking = st.button(
            "▶️ Start Video Tracking",
            type="primary"
        )

        if start_tracking:

            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():

                st.error("❌ Could not open the uploaded video.")

            else:

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                st.subheader("🚘 Live Video Tracking")

                frame_placeholder = st.empty()

                progress_bar = st.progress(0)

                status = st.empty()

                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

                col1, col2, col3, col4, col5 = st.columns(5)

                metric_total = col1.empty()
                metric_car = col2.empty()
                metric_motorcycle = col3.empty()
                metric_bus = col4.empty()
                metric_truck = col5.empty()

                frame_number = 0

                max_seen = {
                    "Car": 0,
                    "Motorcycle": 0,
                    "Bus": 0,
                    "Truck": 0
                }

                # ------------------------------------------------
                # PROCESS VIDEO
                # ------------------------------------------------

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    # ------------------------------------------------
                    # YOLO TRACKING
                    #
                    # IMPORTANT:
                    # ByteTrack is used instead of BoT-SORT.
                    # This avoids the previous "No module named lap"
                    # error.
                    # ------------------------------------------------

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        verbose=False
                    )

                    result = results[0]

                    processed_frame, counts, total = draw_tracking(
                        frame,
                        result
                    )

                    # ------------------------------------------------
                    # UPDATE METRICS
                    # ------------------------------------------------

                    for key in max_seen:
                        max_seen[key] = max(
                            max_seen[key],
                            counts[key]
                        )

                    metric_total.metric(
                        "Current Vehicles",
                        total
                    )

                    metric_car.metric(
                        "Cars",
                        counts["Car"]
                    )

                    metric_motorcycle.metric(
                        "Motorcycles",
                        counts["Motorcycle"]
                    )

                    metric_bus.metric(
                        "Buses",
                        counts["Bus"]
                    )

                    metric_truck.metric(
                        "Trucks",
                        counts["Truck"]
                    )

                    # ------------------------------------------------
                    # DISPLAY FRAME
                    # ------------------------------------------------

                    processed_rgb = cv2.cvtColor(
                        processed_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_placeholder.image(
                        processed_rgb,
                        channels="RGB",
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # PROGRESS
                    # ------------------------------------------------

                    if total_frames > 0:

                        progress = min(
                            frame_number / total_frames,
                            1.0
                        )

                        progress_bar.progress(progress)

                    status.write(
                        f"Processing frame {frame_number}"
                        f" / {total_frames}"
                    )

                    # ------------------------------------------------
                    # CONTROL PLAYBACK SPEED
                    # ------------------------------------------------

                    time.sleep(
                        max(0.001, 1 / fps)
                    )

                cap.release()

                progress_bar.progress(1.0)

                st.success(
                    "✅ Video tracking completed successfully!"
                )

                st.subheader("📊 Maximum Vehicles Detected")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Cars",
                    max_seen["Car"]
                )

                c2.metric(
                    "Motorcycles",
                    max_seen["Motorcycle"]
                )

                c3.metric(
                    "Buses",
                    max_seen["Bus"]
                )

                c4.metric(
                    "Trucks",
                    max_seen["Truck"]
                )

        # ----------------------------------------------------
        # DELETE TEMP FILE
        # ----------------------------------------------------

        try:
            os.unlink(temp_input.name)
        except Exception:
            pass


# ============================================================
# LIVE WEBCAM
# ============================================================

else:

    st.header("📹 Live Webcam Tracking")

    st.info(
        "Allow camera permission when your browser asks."
    )

    # --------------------------------------------------------
    # Import WebRTC only in webcam mode
    # --------------------------------------------------------

    try:

        import av

        from streamlit_webrtc import (
            webrtc_streamer,
            VideoProcessorBase,
            RTCConfiguration
        )

    except Exception as e:

        st.error(
            "Webcam dependencies are not installed."
        )

        st.code(
            "streamlit-webrtc==0.77.0\n"
            "av==14.2.0"
        )

        st.warning(
            f"Dependency error: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # THREAD LOCK
    # --------------------------------------------------------

    model_lock = threading.Lock()


    # --------------------------------------------------------
    # WEBRTC CONFIG
    # --------------------------------------------------------

    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )


    # --------------------------------------------------------
    # VIDEO PROCESSOR
    # --------------------------------------------------------

    class VehicleVideoProcessor(VideoProcessorBase):

        def __init__(self):

            self.vehicle_count = 0

        def recv(self, frame):

            img = frame.to_ndarray(
                format="bgr24"
            )

            try:

                with model_lock:

                    results = model.track(
                        img,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        verbose=False
                    )

                result = results[0]

                processed, counts, total = draw_tracking(
                    img,
                    result
                )

                self.vehicle_count = total

            except Exception as e:

                cv2.putText(
                    img,
                    f"Tracking error: {str(e)[:60]}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                processed = img

            return av.VideoFrame.from_ndarray(
                processed,
                format="bgr24"
            )


    # --------------------------------------------------------
    # START WEBCAM
    # --------------------------------------------------------

    webrtc_ctx = webrtc_streamer(
        key="vehicle-live-tracking",
        video_processor_factory=VehicleVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


    st.markdown("---")

    st.subheader("🚗 Live Tracking")

    if webrtc_ctx.state.playing:

        st.success(
            "🟢 Live vehicle tracking is running."
        )

        st.write(
            "The YOLO model is detecting and tracking "
            "vehicles from your webcam."
        )

    else:

        st.warning(
            "Click START above and allow camera permission."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🚗 AI Vehicle Detection & Tracking | "
    "YOLOv8 + ByteTrack + Streamlit"
)
