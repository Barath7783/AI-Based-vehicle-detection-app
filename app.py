import streamlit as st
import cv2
import tempfile
import threading
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection, Tracking & Counting")

st.caption(
    "YOLOv8 + ByteTrack | Live Webcam & Video Upload"
)


# =========================================================
# VEHICLE CLASSES
# =========================================================

# COCO dataset classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# =========================================================
# TRACKING LOCK
# =========================================================

model_lock = threading.Lock()


# =========================================================
# LIVE WEBCAM CALLBACK
# =========================================================

def live_video_callback(frame):

    # Convert WebRTC frame to OpenCV format
    image = frame.to_ndarray(format="bgr24")

    # Run YOLO tracking
    with model_lock:

        results = model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=0.30,
            verbose=False
        )

    result = results[0]

    # Draw YOLO boxes
    output = result.plot()

    current_ids = []

    # -----------------------------------------------------
    # GET TRACK IDs
    # -----------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        for box, track_id, cls in zip(
            boxes,
            track_ids,
            classes
        ):

            if cls not in VEHICLE_CLASSES:
                continue

            current_ids.append(track_id)

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # Draw tracking ID
            cv2.putText(
                output,
                f"ID: {track_id}",
                (
                    x1,
                    max(y1 - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    # -----------------------------------------------------
    # DISPLAY INFORMATION
    # -----------------------------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (360, 95),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        f"Vehicles: {len(current_ids)}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Tracking IDs: {len(current_ids)}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Return frame to browser
    return av.VideoFrame.from_ndarray(
        output,
        format="bgr24"
    )


# =========================================================
# SIDEBAR
# =========================================================

option = st.sidebar.radio(
    "Select Input",
    [
        "Live Webcam",
        "Upload Video"
    ]
)


# =========================================================
# LIVE WEBCAM
# =========================================================

if option == "Live Webcam":

    st.header("📷 Live Vehicle Tracking")

    st.info(
        "Click START, select your webcam and allow camera permission."
    )

    st.markdown(
        """
        **Tracking features**

        - 🚗 Car detection
        - 🏍️ Motorcycle detection
        - 🚌 Bus detection
        - 🚚 Truck detection
        - 🆔 Persistent tracking IDs
        """
    )

    # -----------------------------------------------------
    # WEBRTC
    # -----------------------------------------------------

    webrtc_streamer(
        key="vehicle-live-tracking",

        video_frame_callback=live_video_callback,

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
        "If the camera does not appear, select the camera device "
        "and allow browser camera permission."
    )


# =========================================================
# UPLOAD VIDEO
# =========================================================

else:

    st.header("🎥 Vehicle Tracking from Video")

    uploaded_file = st.file_uploader(
        "Choose a vehicle video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_file is not None:

        # -------------------------------------------------
        # ORIGINAL VIDEO
        # -------------------------------------------------

        st.subheader("▶️ Original Video")

        video_bytes = uploaded_file.getvalue()

        st.video(video_bytes)

        st.markdown("---")

        # -------------------------------------------------
        # SAVE TEMPORARY VIDEO
        # -------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(video_bytes)
        input_file.close()

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            input_file.name
        )

        if not cap.isOpened():

            st.error(
                "❌ Unable to open the uploaded video."
            )

            st.stop()

        # -------------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

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

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # -------------------------------------------------
        # OUTPUT VIDEO
        # -------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        st.subheader(
            "🤖 Live Processing"
        )

        frame_placeholder = st.empty()

        metric_col1, metric_col2 = st.columns(2)

        current_metric = metric_col1.empty()

        unique_metric = metric_col2.empty()

        progress = st.progress(0)

        # -------------------------------------------------
        # TRACKING VARIABLES
        # -------------------------------------------------

        all_vehicle_ids = set()

        frame_number = 0

        # -------------------------------------------------
        # PROCESS VIDEO
        # -------------------------------------------------

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # -------------------------------------------------
            # YOLO TRACKING
            # -------------------------------------------------

            with model_lock:

                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=0.30,
                    verbose=False
                )

            result = results[0]

            current_ids = []

            # -------------------------------------------------
            # GET TRACK IDs
            # -------------------------------------------------

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                classes = (
                    result.boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )

                for box, track_id, cls in zip(
                    boxes,
                    track_ids,
                    classes
                ):

                    if cls not in VEHICLE_CLASSES:
                        continue

                    current_ids.append(
                        track_id
                    )

                    all_vehicle_ids.add(
                        track_id
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    # -------------------------------------------------
                    # DRAW TRACK ID
                    # -------------------------------------------------

                    cv2.putText(
                        result.plot(),
                        f"ID: {track_id}",
                        (
                            x1,
                            max(
                                y1 - 10,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

            # -------------------------------------------------
            # DRAW DETECTIONS
            # -------------------------------------------------

            output_frame = result.plot()

            # -------------------------------------------------
            # DRAW TRACK IDS AGAIN
            # -------------------------------------------------

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )

                classes = (
                    result.boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )

                for box, track_id, cls in zip(
                    boxes,
                    track_ids,
                    classes
                ):

                    if cls not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cv2.putText(
                        output_frame,
                        f"ID: {track_id}",
                        (
                            x1,
                            max(
                                y1 - 10,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

            # -------------------------------------------------
            # INFORMATION BOX
            # -------------------------------------------------

            cv2.rectangle(
                output_frame,
                (10, 10),
                (380, 105),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                output_frame,
                f"Current Vehicles: {len(current_ids)}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Unique Vehicles: {len(all_vehicle_ids)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # WRITE OUTPUT
            # -------------------------------------------------

            out.write(
                output_frame
            )

            # -------------------------------------------------
            # SHOW FRAME
            # -------------------------------------------------

            frame_placeholder.image(
                output_frame,
                channels="BGR",
                use_container_width=True
            )

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            current_metric.metric(
                "Vehicles in Current Frame",
                len(current_ids)
            )

            unique_metric.metric(
                "Unique Vehicles Tracked",
                len(all_vehicle_ids)
            )

            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            frame_number += 1

            if total_frames > 0:

                progress_value = (
                    frame_number /
                    total_frames
                )

                progress.progress(
                    min(
                        progress_value,
                        1.0
                    )
                )

        # -------------------------------------------------
        # RELEASE RESOURCES
        # -------------------------------------------------

        cap.release()
        out.release()

        progress.progress(1.0)

        st.success(
            "✅ Vehicle tracking completed!"
        )

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Tracking Results"
        )

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Unique Vehicles",
            len(all_vehicle_ids)
        )

        result_col2.metric(
            "Frames Processed",
            frame_number
        )

        # -------------------------------------------------
        # PROCESSED VIDEO
        # -------------------------------------------------

        st.subheader(
            "🚗 Processed Tracking Video"
        )

        try:

            with open(
                output_path,
                "rb"
            ) as processed_file:

                processed_bytes = (
                    processed_file.read()
                )

            st.video(
                processed_bytes
            )

        except Exception:

            st.warning(
                "Processed video could not be displayed. "
                "The live processed frames above are still available."
            )
