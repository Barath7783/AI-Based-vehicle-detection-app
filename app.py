import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Vehicle Detection & Counting",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚗 Vehicle Detection & Counting")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Select Input")

input_type = st.sidebar.radio(
    "",
    ["Webcam", "Upload Video"]
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# =========================================================
# VEHICLE CLASSES
# =========================================================

CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

VEHICLE_CLASSES = [2, 3, 5, 7]

# =========================================================
# SESSION STATE
# =========================================================

if "tracking" not in st.session_state:
    st.session_state.tracking = False

if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = False

# =========================================================
# UPLOAD VIDEO
# =========================================================

if input_type == "Upload Video":

    st.subheader("🎥 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        # Save video temporarily
        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_file.write(
            uploaded_file.read()
        )

        temp_file.close()

        # =================================================
        # VIDEO INFORMATION
        # =================================================

        cap_info = cv2.VideoCapture(
            temp_file.name
        )

        fps = cap_info.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap_info.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap_info.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap_info.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap_info.release()

        if fps <= 0:
            fps = 25

        # =================================================
        # START / STOP
        # =================================================

        col1, col2 = st.columns(2)

        with col1:
            start = st.button(
                "▶ START",
                type="primary",
                use_container_width=True
            )

        with col2:
            stop = st.button(
                "⏹ STOP",
                use_container_width=True
            )

        if stop:
            st.session_state.stop_tracking = True

        # =================================================
        # VIDEO AREA
        # =================================================

        video_placeholder = st.empty()

        # =================================================
        # STATISTICS
        # =================================================

        stats_placeholder = st.empty()

        progress_placeholder = st.empty()

        # =================================================
        # START TRACKING
        # =================================================

        if start:

            st.session_state.stop_tracking = False

            cap = cv2.VideoCapture(
                temp_file.name
            )

            if not cap.isOpened():

                st.error(
                    "Unable to open video."
                )

                st.stop()

            # ---------------------------------------------
            # UNIQUE TRACKING IDS
            # ---------------------------------------------

            tracked_ids = set()

            car_ids = set()
            motorcycle_ids = set()
            bus_ids = set()
            truck_ids = set()

            frame_number = 0

            # =============================================
            # PROCESS VIDEO
            # =============================================

            while cap.isOpened():

                # -----------------------------------------
                # STOP BUTTON
                # -----------------------------------------

                if st.session_state.stop_tracking:
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                # -----------------------------------------
                # YOLO TRACKING
                #
                # ByteTrack avoids LAP dependency
                # -----------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    iou=0.45,
                    classes=[0, 2, 3, 5, 7],
                    verbose=False
                )

                result = results[0]

                # -----------------------------------------
                # CURRENT COUNTS
                # -----------------------------------------

                current_car = 0
                current_motorcycle = 0
                current_bus = 0
                current_truck = 0
                current_person = 0

                # -----------------------------------------
                # DETECTIONS
                # -----------------------------------------

                if result.boxes is not None:

                    boxes = result.boxes

                    for i in range(
                        len(boxes)
                    ):

                        cls_id = int(
                            boxes.cls[i].item()
                        )

                        if cls_id not in CLASS_NAMES:
                            continue

                        name = CLASS_NAMES[
                            cls_id
                        ]

                        confidence = float(
                            boxes.conf[i].item()
                        )

                        # ---------------------------------
                        # COUNT CURRENT FRAME
                        # ---------------------------------

                        if name == "car":
                            current_car += 1

                        elif name == "motorcycle":
                            current_motorcycle += 1

                        elif name == "bus":
                            current_bus += 1

                        elif name == "truck":
                            current_truck += 1

                        elif name == "person":
                            current_person += 1

                        # ---------------------------------
                        # TRACK ID
                        # ---------------------------------

                        track_id = None

                        if boxes.id is not None:

                            track_id = int(
                                boxes.id[i].item()
                            )

                            tracked_ids.add(
                                track_id
                            )

                            if name == "car":
                                car_ids.add(
                                    track_id
                                )

                            elif name == "motorcycle":
                                motorcycle_ids.add(
                                    track_id
                                )

                            elif name == "bus":
                                bus_ids.add(
                                    track_id
                                )

                            elif name == "truck":
                                truck_ids.add(
                                    track_id
                                )

                        # ---------------------------------
                        # BOUNDING BOX
                        # ---------------------------------

                        x1, y1, x2, y2 = map(
                            int,
                            boxes.xyxy[i].tolist()
                        )

                        # ---------------------------------
                        # LABEL
                        # ---------------------------------

                        if track_id is not None:

                            label = (
                                f"{name} "
                                f"ID:{track_id} "
                                f"{confidence:.2f}"
                            )

                        else:

                            label = (
                                f"{name} "
                                f"{confidence:.2f}"
                            )

                        # ---------------------------------
                        # BOX
                        # ---------------------------------

                        if name == "person":
                            box_color = (
                                255,
                                0,
                                0
                            )

                        elif name == "car":
                            box_color = (
                                255,
                                255,
                                255
                            )

                        elif name == "motorcycle":
                            box_color = (
                                0,
                                255,
                                255
                            )

                        elif name == "bus":
                            box_color = (
                                255,
                                0,
                                255
                            )

                        else:
                            box_color = (
                                0,
                                255,
                                255
                            )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            box_color,
                            2
                        )

                        # ---------------------------------
                        # LABEL BACKGROUND
                        # ---------------------------------

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        (tw, th), baseline = (
                            cv2.getTextSize(
                                label,
                                font,
                                0.55,
                                2
                            )
                        )

                        label_y = max(
                            y1 - 5,
                            th + 5
                        )

                        cv2.rectangle(
                            frame,
                            (
                                x1,
                                label_y - th - 8
                            ),
                            (
                                x1 + tw + 8,
                                label_y + baseline
                            ),
                            box_color,
                            -1
                        )

                        # ---------------------------------
                        # LABEL TEXT
                        # ---------------------------------

                        cv2.putText(
                            frame,
                            label,
                            (
                                x1 + 4,
                                label_y - 4
                            ),
                            font,
                            0.55,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA
                        )

                # =================================================
                # TOTAL VEHICLES IN CURRENT FRAME
                # =================================================

                total_current = (
                    current_car
                    + current_motorcycle
                    + current_bus
                    + current_truck
                )

                # =================================================
                # TOP INFORMATION
                # =================================================

                info = (
                    f"Total Vehicles: "
                    f"{total_current}"
                )

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (330, 60),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    info,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # =================================================
                # CONVERT RGB
                # =================================================

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # =================================================
                # DISPLAY LIVE VIDEO
                # =================================================

                video_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

                # =================================================
                # PROGRESS
                # =================================================

                if total_frames > 0:

                    percentage = (
                        frame_number
                        / total_frames
                    )

                    progress_placeholder.progress(
                        min(
                            percentage,
                            1.0
                        )
                    )

                # =================================================
                # STATISTICS
                # =================================================

                stats_placeholder.markdown(
                    f"""
                    ### 🚗 Vehicle Detection

                    | Vehicle | Current |
                    |---|---:|
                    | 🚗 Car | **{current_car}** |
                    | 🏍️ Motorcycle | **{current_motorcycle}** |
                    | 🚌 Bus | **{current_bus}** |
                    | 🚚 Truck | **{current_truck}** |
                    | 👤 Person | **{current_person}** |

                    **Total Vehicles in Current Frame: {total_current}**

                    **Unique Tracked Vehicles: {len(tracked_ids)}**
                    """
                )

                # =================================================
                # PLAYBACK SPEED
                # =================================================

                time.sleep(
                    max(
                        0.001,
                        1.0 / fps
                    )
                )

            # =================================================
            # RELEASE VIDEO
            # =================================================

            cap.release()

            if st.session_state.stop_tracking:

                st.warning(
                    "⏹ Tracking stopped."
                )

            else:

                st.success(
                    "✅ Video tracking completed!"
                )

            # =================================================
            # FINAL RESULTS
            # =================================================

            st.subheader(
                "📊 Final Tracking Results"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "🚗 Cars",
                    len(car_ids)
                )

            with c2:
                st.metric(
                    "🏍️ Motorcycles",
                    len(motorcycle_ids)
                )

            with c3:
                st.metric(
                    "🚌 Buses",
                    len(bus_ids)
                )

            with c4:
                st.metric(
                    "🚚 Trucks",
                    len(truck_ids)
                )

else:

    # =========================================================
    # WEBCAM
    # =========================================================

    st.subheader(
        "📷 Webcam"
    )

    st.info(
        "Webcam live tracking can be enabled "
        "when running the app locally. "
        "For Streamlit Cloud, use Upload Video."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Based Vehicle Detection & Counting • "
    "YOLOv8 + ByteTrack + Streamlit"
)
