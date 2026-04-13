import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="Vehicle Detection", layout="wide")

st.title("🚗 Vehicle Detection & Counting")

# Load YOLO model
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]  # car, bike, bus, truck

# Sidebar
option = st.sidebar.radio("Select Input", ["Webcam", "Upload Video"])

frame_placeholder = st.empty()
count_placeholder = st.empty()

# ---------------- WEBCAM ----------------
if option == "Webcam":
    run = st.sidebar.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            results = model(frame)

            count = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in vehicle_classes and conf > 0.4:
                    count += 1

            frame = results[0].plot()

            cv2.putText(frame, f"Vehicles: {count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            frame_placeholder.image(frame, channels="BGR")
            count_placeholder.metric("Total Vehicles", count)

        cap.release()

# ---------------- VIDEO UPLOAD ----------------
else:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            count = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in vehicle_classes and conf > 0.4:
                    count += 1

            frame = results[0].plot()

            cv2.putText(frame, f"Vehicles: {count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            frame_placeholder.image(frame, channels="BGR")
            count_placeholder.metric("Total Vehicles", count)

        cap.release()