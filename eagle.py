# eagle_streamlit_final.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import threading
import queue
import os
import time

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    status_text = st.markdown("**🔍 Last Detected Status:** None")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    CROWD_THRESHOLD = st.slider("Crowd Size Threshold", 2, 10, 3)
    FRAME_INTERVAL = st.slider("Frame Interval (seconds)", 0.5, 5.0, 1.0)

# -----------------------------
# Load Models
# -----------------------------
with st.spinner("Loading models..."):
    yolo_model = YOLO("yolov8n.pt")
    
    chair_model_path = r"C:\Users\Lenovo\Desktop\FINAL\best.pt"
    chair_model = YOLO(chair_model_path) if os.path.exists(chair_model_path) else yolo_model

    gender_model_path = r"C:\Users\Lenovo\Desktop\FINAL\gender_classifier_model.h5"
    if not os.path.exists(gender_model_path):
        st.error(f"Gender model not found at {gender_model_path}")
        st.stop()
    gender_model = load_model(gender_model_path)

# -----------------------------
# Colors for misconduct
# -----------------------------
MISCONDUCT_COLORS = {
    "Using Phone": (0, 255, 255),
    "Crowding": (0, 0, 255),
    "Loitering": (128, 0, 128),
    "Occupied": (0, 255, 0),
}

# -----------------------------
# Helper functions
# -----------------------------
def predict_gender(face_crop):
    if face_crop.size == 0 or face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
        return "Unknown", 0.0
    try:
        face = cv2.resize(face_crop, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        pred = gender_model.predict(face, verbose=0)[0][0]
        gender = "Female" if pred >= 0.5 else "Male"
        return gender, pred
    except:
        return "Unknown", 0.0

def detect_occupancy(frame):
    chair_result = chair_model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]
    if chair_result.boxes and len(chair_result.boxes.cls) > 0:
        return "Occupied"
    fallback_result = yolo_model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
    labels = fallback_result.boxes.cls.tolist() if fallback_result.boxes else []
    names = yolo_model.names
    person_detected = any(names[int(cls)] == "person" for cls in labels)
    return "Occupied" if person_detected else "Loitering"

def process_frame(frame):
    misconducts_detected = []
    phone_detected = False
    crowd_detected = False

    try:
        results = yolo_model(frame, conf=0.5, imgsz=640, verbose=False)[0]
        person_count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "person":
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_crop = frame[y1:y2, x1:x2]
                gender, confidence = predict_gender(face_crop)
                cv2.putText(frame, f"{gender} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif label == "cell phone":
                phone_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), MISCONDUCT_COLORS["Using Phone"], 2)
                cv2.putText(frame, "Using Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, MISCONDUCT_COLORS["Using Phone"], 2)
        if person_count >= CROWD_THRESHOLD:
            crowd_detected = True
    except:
        pass

    chair_status = detect_occupancy(frame)

    if phone_detected:
        misconducts_detected.append("Using Phone")
    if crowd_detected:
        misconducts_detected.append("Crowding")
    if chair_status == "Loitering":
        misconducts_detected.append("Loitering")
    else:
        misconducts_detected.append("Occupied")

    # Draw labels without overlapping
    for i, misconduct in enumerate(misconducts_detected):
        color = MISCONDUCT_COLORS.get(misconduct, (255, 255, 255))
        cv2.putText(frame, misconduct, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, misconducts_detected

# -----------------------------
# Threads for webcam & processing
# -----------------------------
frame_queue = queue.Queue(maxsize=5)
results_queue = queue.Queue(maxsize=1)
stop_thread = threading.Event()

def video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        return
    try:
        while not stop_thread.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640, 480))
            if frame_queue.full():
                _ = frame_queue.get_nowait()
            frame_queue.put(frame)
    finally:
        cap.release()

def frame_processor():
    last_time = 0
    while not stop_thread.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        # Throttle FPS
        current_time = time.time()
        if current_time - last_time < FRAME_INTERVAL:
            continue
        last_time = current_time

        processed_frame, misconducts = process_frame(frame)
        if results_queue.full():
            _ = results_queue.get_nowait()
        results_queue.put((processed_frame, misconducts))

# Start threads
threading.Thread(target=video_capture, daemon=True).start()
threading.Thread(target=frame_processor, daemon=True).start()

# -----------------------------
# Streamlit display loop
# -----------------------------
try:
    while True:
        try:
            frame, misconducts = results_queue.get(timeout=1)
        except queue.Empty:
            time.sleep(0.01)
            continue

        status_text.markdown("**🔍 Last Detected Status:** " + " | ".join(misconducts))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(Image.fromarray(img))

finally:
    stop_thread.set()
