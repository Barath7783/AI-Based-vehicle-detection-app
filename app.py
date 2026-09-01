import os
import tempfile
from pathlib import Path

import cv2
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ============================================================
# PAGE CONFIGURATION
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
st.caption("YOLO-based vehicle detection, counting and live tracking")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

MODEL_PATH = Path("yolov8n.pt")

if not MODEL_PATH.exists():
    st.error(
        "❌ yolov8n.pt not found. "
        "Please upload yolov8n.pt to the root of your GitHub repository."
    )
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


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
# SIDEBAR
# ============================================================

st.sidebar.header("Select Input")

option = st.sidebar.radio(
    "Input Type",
    [
        "Live Webcam",
        "Upload Video"
    ]
)


# ============================================================
# LIVE WEBCAM PROCESSOR
# ============================================================

class VehicleTracker(VideoProcessorBase):

    def __init__(self):
        self.model = load_model()

        self.total_current = 0
        self.car_count = 0
        self.motorcycle_count = 0
        self.bus_count = 0
        self.truck_count = 0

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV
        img = frame.to_ndarray(format="bgr24")

        try:

            # YOLO TRACKING
            results = self.model.track(
                img,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                conf=0.35,
                verbose=False
            )

            result = results[0]

            # Reset current counts
            self.total_current = 0
            self.car_count = 0
            self.motorcycle_count = 0
            self.bus_count = 0
            self.truck_count = 0

            if result.boxes is not None:

                for box in result.boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls not in VEHICLE_CLASSES:
                        continue

                    self.total_current += 1

                    # Vehicle type count
                    if cls == 2:
                        self.car_count += 1

                    elif cls == 3:
                        self.motorcycle_count += 1

                    elif cls == 5:
                        self.bus_count += 1

                    elif cls == 7:
                        self.truck_count += 1

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    # Tracking ID
                    track_id = None

                    if box.id is not None:
                        track_id = int(box.id[0])

                    label = VEHICLE_CLASSES[cls]

                    if track_id is not None:
                        text = f"{label} ID:{track_id} {conf:.2f}"
                    else:
                        text = f"{label} {conf:.2f}"

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Text background
                    (tw, th), _ = cv2.getTextSize(
                        text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2
                    )

                    cv2.rectangle(
                        img,
                        (x1, max(0, y1 - th - 10)),
                        (x1 + tw + 5, y1),
                        (0, 255, 0),
                        -1
                    )

                    # Label
                    cv2.putText(
                        img,
                        text,
                        (x1 + 2, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2
                    )

            # =================================================
            # TOP INFORMATION PANEL
            # =================================================

            cv2.rectangle(
                img,
                (10, 10),
                (350, 145),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                img,
                f"Vehicles: {self.total_current}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Cars: {self.car_count}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Motorcycles: {self.motorcycle_count}",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Bus: {self.bus_count}  Truck: {self.truck_count}",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        except Exception as e:

            cv2.putText(
                img,
                f"Tracking Error: {str(e)[:60]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# LIVE WEBCAM
# ============================================================

if option == "Live Webcam":

    st.subheader("📹 Live Vehicle Tracking")

    st.info(
        "Click START and allow camera permission in your browser."
    )

    # WebRTC configuration
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

    ctx = webrtc_streamer(
        key="vehicle-tracking-webcam",
        video_processor_factory=VehicleTracker,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown("---")

    st.subheader("📊 Live Detection")

    col1, col2, col3, col4 = st.columns(4)

    if ctx.video_processor:

        processor = ctx.video_processor

        with col1:
            st.metric(
                "Vehicles",
                processor.total_current
            )

        with col2:
            st.metric(
                "Cars",
                processor.car_count
            )

        with col3:
            st.metric(
                "Motorcycles",
                processor.motorcycle_count
            )

        with col4:
            st.metric(
                "Bus / Truck",
                processor.bus_count + processor.truck_count
            )

    else:

        with col1:
            st.metric("Vehicles", 0)

        with col2:
            st.metric("Cars", 0)

        with col3:
            st.metric("Motorcycles", 0)

        with col4:
            st.metric("Bus / Truck", 0)


# ============================================================
# UPLOAD VIDEO
# ============================================================

else:

    st.subheader("🎥 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_file is not None:

        # Save uploaded video temporarily
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            video_path = temp_file.name

        st.success(
            f"Video uploaded: {uploaded_file.name}"
        )

        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            st.error(
                "❌ Unable to open uploaded video."
            )

            st.stop()

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )

        st.write(
            f"**Resolution:** {width} × {height}"
        )

        st.write(
            f"**FPS:** {fps:.2f}"
        )

        st.write(
            f"**Duration:** {duration:.1f} seconds"
        )

        st.markdown("---")

        # ====================================================
        # OUTPUT AREA
        # ====================================================

        st.subheader("🤖 Vehicle Tracking Output")

        video_placeholder = st.empty()

        progress_bar = st.progress(0)

        count_placeholder = st.empty()

        # Current frame count
        frame_number = 0

        # Maximum unique tracking IDs
        unique_ids = set()

        # Vehicle type totals
        total_cars = 0
        total_motorcycles = 0
        total_buses = 0
        total_trucks = 0

        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            try:

                # YOLO tracking
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.35,
                    verbose=False
                )

                result = results[0]

                current_count = 0

                current_cars = 0
                current_motorcycles = 0
                current_buses = 0
                current_trucks = 0

                if result.boxes is not None:

                    for box in result.boxes:

                        cls = int(
                            box.cls[0]
                        )

                        conf = float(
                            box.conf[0]
                        )

                        if cls not in VEHICLE_CLASSES:
                            continue

                        current_count += 1

                        # ====================================
                        # VEHICLE TYPE
                        # ====================================

                        if cls == 2:
                            current_cars += 1

                        elif cls == 3:
                            current_motorcycles += 1

                        elif cls == 5:
                            current_buses += 1

                        elif cls == 7:
                            current_trucks += 1

                        # ====================================
                        # TRACKING ID
                        # ====================================

                        track_id = None

                        if box.id is not None:

                            track_id = int(
                                box.id[0]
                            )

                            unique_ids.add(
                                track_id
                            )

                        # ====================================
                        # BOUNDING BOX
                        # ====================================

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        label = VEHICLE_CLASSES[cls]

                        if track_id is not None:

                            text = (
                                f"{label} "
                                f"ID:{track_id} "
                                f"{conf:.2f}"
                            )

                        else:

                            text = (
                                f"{label} "
                                f"{conf:.2f}"
                            )

                        # Draw box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        # Text size
                        (tw, th), _ = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            2
                        )

                        # Text background
                        cv2.rectangle(
                            frame,
                            (
                                x1,
                                max(
                                    0,
                                    y1 - th - 10
                                )
                            ),
                            (
                                x1 + tw + 5,
                                y1
                            ),
                            (0, 255, 0),
                            -1
                        )

                        # Text
                        cv2.putText(
                            frame,
                            text,
                            (
                                x1 + 2,
                                max(
                                    15,
                                    y1 - 5
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 0),
                            2
                        )

                # =================================================
                # TOP INFORMATION PANEL
                # =================================================

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (390, 175),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Vehicles: {current_count}",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Unique IDs: {len(unique_ids)}",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Cars: {current_cars}",
                    (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Motorcycles: {current_motorcycles}",
                    (25, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Bus: {current_buses}  Truck: {current_trucks}",
                    (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # =================================================
                # SHOW FRAME
                # =================================================

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                video_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

                # =================================================
                # COUNTERS
                # =================================================

                with count_placeholder.container():

                    c1, c2, c3, c4, c5 = st.columns(5)

                    c1.metric(
                        "Current Vehicles",
                        current_count
                    )

                    c2.metric(
                        "Unique Vehicles",
                        len(unique_ids)
                    )

                    c3.metric(
                        "Cars",
                        current_cars
                    )

                    c4.metric(
                        "Motorcycles",
                        current_motorcycles
                    )

                    c5.metric(
                        "Bus + Truck",
                        current_buses + current_trucks
                    )

                # =================================================
                # PROGRESS
                # =================================================

                if total_frames > 0:

                    progress = min(
                        frame_number / total_frames,
                        1.0
                    )

                    progress_bar.progress(
                        progress
                    )

            except Exception as e:

                st.error(
                    f"Tracking error: {e}"
                )

                break

        # ====================================================
        # RELEASE VIDEO
        # ====================================================

        cap.release()

        progress_bar.progress(1.0)

        st.success(
            "✅ Video processing completed!"
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Tracking Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Unique Vehicles Tracked",
                len(unique_ids)
            )

        with col2:
            st.metric(
                "Frames Processed",
                frame_number
            )

        with col3:
            st.metric(
                "Video Duration",
                f"{duration:.1f}s"
            )

        # Delete temporary video
        try:
            os.unlink(video_path)
        except Exception:
            pass
