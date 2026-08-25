import streamlit as st
import cv2
import tempfile
import threading
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection, Tracking & Counting")


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# COCO classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]


# =========================================================
# SHARED STATE FOR LIVE WEBCAM
# =========================================================

lock = threading.Lock()

live_state = {
    "count": 0,
    "ids": set(),
    "total": 0
}


# =========================================================
# LIVE WEBCAM TRACKING
# =========================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(format="bgr24")

    # YOLO TRACKING
    results = model.track(
        image,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        conf=0.4,
        verbose=False
    )

    result = results[0]

    current_ids = []

    # -----------------------------------------------------
    # GET TRACK IDs
    # -----------------------------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        classes = result.boxes.cls.int().cpu().tolist()

        confidences = result.boxes.conf.cpu().tolist()

        for track_id, cls, conf in zip(
            track_ids,
            classes,
            confidences
        ):

            if cls in VEHICLE_CLASSES and conf >= 0.4:

                current_ids.append(track_id)

    # -----------------------------------------------------
    # DRAW TRACKING
    # -----------------------------------------------------

    output = result.plot()

    # -----------------------------------------------------
    # DISPLAY TRACK IDs
    # -----------------------------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().tolist()

        track_ids = result.boxes.id.int().cpu().tolist()

        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(
            boxes,
            track_ids,
            classes
        ):

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            cv2.putText(
                output,
                f"ID: {track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------------------------------
    # UPDATE SHARED STATE
    # -----------------------------------------------------

    with lock:

        live_state["count"] = len(current_ids)

        live_state["ids"] = set(current_ids)

        live_state["total"] = max(
            live_state["total"],
            len(live_state["ids"])
        )

    # -----------------------------------------------------
    # DISPLAY COUNTER
    # -----------------------------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (330, 80),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        f"Vehicles: {len(current_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Tracking IDs: {len(set(current_ids))}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

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

    st.subheader("📷 Live Vehicle Tracking")

    st.info(
        "Click START and allow camera permission."
    )

    ctx = webrtc_streamer(
        key="vehicle-live-tracking",

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

    # -----------------------------------------------------
    # LIVE STATISTICS
    # -----------------------------------------------------

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        current_placeholder = st.empty()

    with col2:

        tracking_placeholder = st.empty()

    if ctx.state.playing:

        while ctx.state.playing:

            with lock:

                current_count = live_state["count"]

                active_ids = len(
                    live_state["ids"]
                )

            current_placeholder.metric(
                "Vehicles in Current Frame",
                current_count
            )

            tracking_placeholder.metric(
                "Active Tracking IDs",
                active_ids
            )


# =========================================================
# UPLOAD VIDEO TRACKING
# =========================================================

else:

    st.subheader("🎥 Vehicle Tracking from Video")

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

        st.subheader("▶️ Original Video")

        st.video(uploaded_file)

        # -------------------------------------------------
        # SAVE VIDEO
        # -------------------------------------------------

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(
            uploaded_file.getbuffer()
        )

        tfile.close()

        cap = cv2.VideoCapture(
            tfile.name
        )

        if not cap.isOpened():

            st.error(
                "Unable to open video."
            )

        else:

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

            frame_placeholder = st.empty()

            count_placeholder = st.empty()

            progress = st.progress(0)

            frame_number = 0

            all_vehicle_ids = set()

            # -------------------------------------------------
            # TRACK VIDEO
            # -------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                # IMPORTANT:
                # persist=True keeps IDs between frames

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=0.4,
                    verbose=False
                )

                result = results[0]

                current_ids = []

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

                    for track_id, cls in zip(
                        track_ids,
                        classes
                    ):

                        if cls in VEHICLE_CLASSES:

                            current_ids.append(
                                track_id
                            )

                            all_vehicle_ids.add(
                                track_id
                            )

                # -------------------------------------------------
                # DRAW TRACKING
                # -------------------------------------------------

                output_frame = result.plot()

                # -------------------------------------------------
                # DRAW ID LABELS
                # -------------------------------------------------

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):

                    boxes = (
                        result.boxes.xyxy
                        .cpu()
                        .tolist()
                    )

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
                                    20
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                # -------------------------------------------------
                # COUNTER
                # -------------------------------------------------

                cv2.rectangle(
                    output_frame,
                    (10, 10),
                    (350, 85),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output_frame,
                    f"Current: {len(current_ids)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    output_frame,
                    f"Unique IDs: {len(all_vehicle_ids)}",
                    (20, 70),
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

                count_placeholder.metric(
                    "Unique Vehicles Tracked",
                    len(all_vehicle_ids)
                )

                frame_number += 1

                if total_frames > 0:

                    progress.progress(
                        min(
                            frame_number /
                            total_frames,
                            1.0
                        )
                    )

            cap.release()
            out.release()

            progress.progress(1.0)

            st.success(
                "✅ Tracking completed!"
            )

            # -------------------------------------------------
            # PROCESSED VIDEO
            # -------------------------------------------------

            st.subheader(
                "🚗 Tracked Video"
            )

            with open(
                output_path,
                "rb"
            ) as video_file:

                processed_video = (
                    video_file.read()
                )

            st.video(
                processed_video
            )

            st.metric(
                "Unique Vehicles Tracked",
                len(all_vehicle_ids)
            )
