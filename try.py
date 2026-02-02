import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import threading
import queue
import os

st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

# Layout
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    status_text = st.markdown("**🔍 Last Detected Status:** None")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    MOUTH_OPEN_THRESHOLD = st.slider("Mouth Open Threshold", 0.05, 0.5, 0.15, 0.05)
    EATING_FRAMES = st.slider("Eating Frames Threshold", 5, 60, 8, 1)
    EYE_CLOSURE_THRESHOLD = st.slider("Eye Closure Threshold (EAR)", 0.1, 0.5, 0.22, 0.01)
    SLEEP_FRAMES = st.slider("Sleep Frames Threshold", 20, 60, 40, 1)
    HEAD_DOWN_THRESHOLD = st.slider("Head Down Threshold", 0.05, 0.5, 0.15, 0.05)
    CROWD_THRESHOLD = st.slider("Crowd Size Threshold", 2, 10, 3, 1)

# Load models
try:
    with st.spinner("Loading models..."):
        yolo_model = YOLO("yolov8n.pt")

        chair_model_path = r"C:\Users\Lenovo\Desktop\FINAL\best.pt"
        if not os.path.exists(chair_model_path):
            st.error(f"Chair model not found at {chair_model_path}. Using fallback YOLOv8n model.")
            chair_model = YOLO("yolov8n.pt")
        else:
            chair_model = YOLO(chair_model_path)

        gender_model_path = r"C:\Users\Lenovo\Desktop\FINAL\gender_classifier_model.h5"
        if not os.path.exists(gender_model_path):
            st.error(f"Gender model not found at {gender_model_path}")
            st.stop()
        gender_model = load_model(gender_model_path)

        # OpenCV cascades instead of Mediapipe
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# State variables
closed_counter = 0
eating_counter = 0
ear_queue = deque(maxlen=5)
mouth_state_queue = deque(maxlen=10)

BOX_COLORS = {
    "Sleeping": (0, 0, 255),
    "Eating": (0, 255, 0),
    "Using Phone": (255, 255, 0),
    "Head Bending": (128, 0, 128),
    "Crowding": (255, 0, 0)
}
DEFAULT_BOX_COLOR = (255, 255, 255)

frame_queue = queue.Queue(maxsize=10)
stop_thread = threading.Event()

def video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
        return
    try:
        while not stop_thread.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640, 480))
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    finally:
        cap.release()

thread = threading.Thread(target=video_capture, daemon=True)
thread.start()

def predict_gender(face_crop, prev_face_crop=None):
    if face_crop.size == 0:
        return "Unknown", 0.0
    try:
        face = cv2.resize(face_crop, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        pred = gender_model.predict(face, verbose=0)[0][0]
        gender = "Female" if pred < 0.5 else "Male"
        return gender, pred
    except:
        return "Unknown", 0.0

def detect_occupancy(frame):
    chair_result = chair_model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]
    if chair_result.boxes and len(chair_result.boxes.cls) > 0:
        return "Occupied"
    fallback_result = yolo_model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
    names = yolo_model.names
    labels = fallback_result.boxes.cls.tolist() if fallback_result.boxes else []
    person_detected = any(names[int(cls)] == "person" for cls in labels)
    return "Occupied" if person_detected else "Loitering!!!!"

frame_count = 0
prev_gender = "Unknown"
prev_confidence = 0.0
last_ui_update = 0

try:
    while not stop_thread.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue

        frame_count += 1
        h, w = frame.shape[:2]
        sleep_status = "Awake"
        eating_status = ""
        phone_detected = False
        crowd_detected = False
        chair_status = "Loitering!!!!"
        gender = prev_gender
        confidence = prev_confidence
        behavior = "Awake"
        box_color = DEFAULT_BOX_COLOR

        # YOLO detection
        if frame_count % 2 == 0:
            try:
                yolo_results = yolo_model(frame, conf=0.5, imgsz=640, verbose=False)[0]
                person_count = 0
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_results.names[cls_id]
                    if label.lower() == "person":
                        person_count += 1
                    elif label.lower() == "cell phone":
                        phone_detected = True
                if person_count >= CROWD_THRESHOLD:
                    crowd_detected = True
            except:
                pass

        # Chair occupancy
        chair_status = detect_occupancy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            for (x, y, w_face, h_face) in faces:
                roi_gray = gray[y:y+h_face, x:x+w_face]
                roi_color = frame[y:y+h_face, x:x+w_face]

                # Eye detection
                eyes = eye_cascade.detectMultiScale(roi_gray)
                ear = 0.25 if len(eyes) >= 2 else 0.1
                ear_queue.append(ear)
                smoothed_ear = np.mean(ear_queue)
                if smoothed_ear < EYE_CLOSURE_THRESHOLD:
                    closed_counter += 1
                    if closed_counter > SLEEP_FRAMES:
                        sleep_status = "Sleeping"
                else:
                    closed_counter = 0

                # Mouth detection
                mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
                mar = 0.2 if len(mouths) > 0 else 0.05
                mouth_state_queue.append(1 if mar > MOUTH_OPEN_THRESHOLD else 0)
                eating_counter += 1 if len(mouths) > 0 else -1
                eating_status = "Eating" if eating_counter > EATING_FRAMES else ""

                # Gender prediction
                gender, confidence = predict_gender(roi_color)
                prev_gender, prev_confidence = gender, confidence

                # Combine results
                misconducts = []
                if sleep_status == "Sleeping":
                    misconducts.append("Sleeping")
                if eating_status:
                    misconducts.append("Eating")
                if phone_detected:
                    misconducts.append("Using Phone")
                if crowd_detected:
                    misconducts.append("Crowding")

                dominant_misconduct = misconducts[0] if misconducts else "Awake"
                box_color = BOX_COLORS.get(dominant_misconduct, DEFAULT_BOX_COLOR)

                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), box_color, 2)
                cv2.putText(frame, f"{dominant_misconduct} | {gender} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            behavior = "Awake"

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
        frame_placeholder.image(img_pil, channels="RGB")
        status_text.markdown(f"**🔍 Last Detected Status:** {behavior} | Chair: {chair_status} | {gender} (Confidence: {confidence:.2f})")

finally:
    stop_thread.set()
    thread.join()
