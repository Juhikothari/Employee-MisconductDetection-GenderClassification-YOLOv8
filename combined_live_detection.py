

# --- crowd.py ---

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
import threading
import queue
import logging
import pandas as pd
import os
from datetime import datetime

# ─── Setup Logging ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Setup CSV Logging ─────────────────────────────────────────────────
LOG_FILE = "misconduct_log.csv"
log_buffer = []

def log_misconduct(timestamp, gender, behavior):
    """Log misconduct events to a CSV file."""
    if behavior != "Awake":
        log_buffer.append({
            "Timestamp": timestamp,
            "Gender": gender,
            "Behavior": behavior
        })
        if len(log_buffer) >= 10:
            file_exists = os.path.isfile(LOG_FILE)
            df = pd.DataFrame(log_buffer)
            df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
            log_buffer.clear()

# ─── Setup Streamlit ───────────────────────────────────────────────────
st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    status_text = st.markdown("**🔍 Last Detected Status:** None")
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        st.subheader("Recent Misconduct Logs")
        st.dataframe(log_df.tail(5), use_container_width=True)

with st.sidebar:
    st.header("Settings")
    st.markdown("""
    **Eye Closure Detection**:
    - Detects closed eyes for sleeping using Eye Aspect Ratio (EAR).
    - Adjust threshold for glasses or lighting.
    """)
    EYE_CLOSURE_THRESHOLD = st.slider("Eye Closure Threshold (EAR)", 0.1, 0.5, 0.22, 0.01,
                                      help="Lower for stricter detection, e.g., for glasses.")
    SLEEP_FRAMES = st.slider("Sleep Frames Threshold", 20, 60, 40, 1,
                             help="Frames with closed eyes to detect sleeping.")
    st.markdown("""
    **Crowd Detection**:
    - Detects gatherings of multiple people.
    - Set minimum number of people for crowding.
    """)
    CROWD_THRESHOLD = st.slider("Crowd Size Threshold", 2, 10, 3, 1,
                                help="Number of people to flag as crowding.")
    st.markdown("""
    **Gender Detection**:
    - Predicts gender from face (0 = Female, 1 = Male, threshold at 0.5).
    - Shows confidence score (0 to 1) for transparency.
    """)

# ─── Load Models ───────────────────────────────────────────────────────
try:
    with st.spinner("Loading models..."):
        yolo_model = YOLO("yolov8n.pt")
        gender_model_path = "C:/Users/Lenovo/Desktop/FINAL/gender_classifier_model.h5"
        if not os.path.exists(gender_model_path):
            st.error(f"Gender model file not found at {gender_model_path}")
            st.stop()
        if not gender_model_path.endswith(('.h5', '.keras')):
            st.error(f"Invalid file format for {gender_model_path}. Keras 3 supports only .h5 or .keras files.")
            st.stop()
        gender_model = load_model(gender_model_path)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ─── Constants ─────────────────────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
FRAME_SKIP = 2
TARGET_RESOLUTION = (640, 480)

# ─── Video Capture Thread ──────────────────────────────────────────────
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
            frame = cv2.resize(frame, TARGET_RESOLUTION)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    finally:
        cap.release()

thread = threading.Thread(target=video_capture, daemon=True)
thread.start()

# ─── Utility Functions ─────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate EAR to measure eye openness."""
    try:
        points = np.array([landmarks[i] for i in eye_indices])
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        return (A + B) / (2.0 * C)
    except:
        return 0.3

def predict_gender(face_crop, prev_face_crop=None):
    """Predict gender with confidence score."""
    if face_crop.size == 0 or face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
        return "Unknown", 0.0
    if prev_face_crop is not None and face_crop.shape == prev_face_crop.shape:
        diff = np.mean(np.abs(face_crop.astype(float) - prev_face_crop.astype(float)))
        if diff < 15:
            return None, 0.0
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

# ─── Main Loop ─────────────────────────────────────────────────────────
ear = 0.3
frame_count = 0
prev_gender = "Unknown"
prev_confidence = 0.0
prev_face_crop = None
last_ui_update = 0
closed_counter = 0
ear_queue = deque(maxlen=5)

try:
    while not stop_thread.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        start_time = time.time()
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sleep_status = "Awake"
        gender = prev_gender
        confidence = prev_confidence
        phone_detected = False
        crowd_detected = False
        behavior = "Awake"

        # ─── Detect Objects with YOLO ───────
        if frame_count % (FRAME_SKIP * 2) == 0:
            try:
                start_yolo = time.time()
                yolo_results = yolo_model(frame, verbose=False)[0]
                person_count = 0
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_results.names[cls_id]
                    conf = box.conf.item()
                    if conf < 0.5:
                        continue
                    if label.lower() == "person":
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    elif label.lower() == "cell phone":
                        phone_detected = True
                if person_count >= CROWD_THRESHOLD:
                    crowd_detected = True
                logger.info(f"YOLO time: {time.time() - start_yolo:.3f}s")
            except Exception as e:
                logger.warning(f"YOLO error: {e}")

        # ─── Detect Face & Sleep ──────────
        start_facemesh = time.time()
        results = face_mesh.process(frame_rgb)
        logger.info(f"FaceMesh time: {time.time() - start_facemesh:.3f}s")

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                ear_queue.append(ear)
                smoothed_ear = np.mean(ear_queue)

                if ear == 0.3 and len(ear_queue) == 5:
                    st.warning("Possible glasses or lighting issue affecting eye detection.")

                if smoothed_ear < EYE_CLOSURE_THRESHOLD:
                    closed_counter += 1
                    if closed_counter > SLEEP_FRAMES:
                        sleep_status = "Sleeping 😴"
                else:
                    closed_counter = 0

                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
                x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)
                face_crop = frame[y1:y2, x1:x2]

                start_gender = time.time()
                gender_pred, confidence_pred = predict_gender(face_crop, prev_face_crop)
                if gender_pred is not None:
                    gender = gender_pred
                    confidence = confidence_pred
                    prev_gender = gender
                    prev_confidence = confidence
                    prev_face_crop = face_crop.copy()
                logger.info(f"Gender time: {time.time() - start_gender:.3f}s")

                # Combine behaviors
                behavior = sleep_status
                if phone_detected:
                    behavior += " + Using Phone"
                if crowd_detected:
                    behavior += " + Crowding"

                # Log misconduct to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_misconduct(timestamp, gender, behavior)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{behavior} | {gender} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            # Handle crowding without face detection
            if crowd_detected:
                behavior = "Crowding"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_misconduct(timestamp, "Unknown", behavior)
            status_text.markdown("**🔍 Last Detected Status:** No face detected (possible sunglasses?)")

        # ─── Display EAR and Frame ─────────
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        current_time = time.time()
        if current_time - last_ui_update >= 0.15:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
            frame_placeholder.image(img_pil, channels="RGB")
            status_text.markdown(f"**🔍 Last Detected Status:** {behavior} | {gender} (Confidence: {confidence:.2f})")
            last_ui_update = current_time

        logger.info(f"Total frame time: {time.time() - start_time:.3f}s")

finally:
    stop_thread.set()
    thread.join()
    face_mesh.close()
    if log_buffer:
        file_exists = os.path.isfile(LOG_FILE)
        df = pd.DataFrame(log_buffer)
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
        log_buffer.clear()

# --- employee.py ---

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# ─── Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

frame_placeholder = st.empty()
status_text = st.markdown("**🔍 Last Detected Status:** None")

# ─── Load Models ───────────────────────────────────────────────────────
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 pretrained model (COCO)
gender_model = load_model("gender_classifier_model.h5")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ─── Constants ─────────────────────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22
SLEEP_FRAMES = 40
closed_counter = 0
ear_queue = deque(maxlen=5)

# ─── Utility Functions ─────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def predict_gender(face_crop):
    try:
        if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
            return "Unknown"
        face = cv2.resize(face_crop, (128, 128))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        pred = gender_model.predict(face, verbose=0)[0][0]
        return "Female" if pred < 0.5 else "Male"
    except:
        return "Unknown"

# ─── Main Loop ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
ear = 0.3  # default EAR

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    sleep_status = "Awake"
    gender = "Unknown"
    phone_detected = False
    behavior = "Awake"

    # ─── Detect Phone with YOLO ───────
    yolo_results = yolo_model(frame, verbose=False)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_results.names[cls_id]
        if label.lower() == "cell phone":
            phone_detected = True
            break

    # ─── Detect Face & Sleep ──────────
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            ear_queue.append(ear)
            smoothed_ear = np.mean(ear_queue)

            if smoothed_ear < EAR_THRESHOLD:
                closed_counter += 1
                if closed_counter > SLEEP_FRAMES:
                    sleep_status = "Sleeping 😴"
            else:
                closed_counter = 0

            # Face Box
            x_coords = [p[0] for p in landmarks]
            y_coords = [p[1] for p in landmarks]
            x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
            x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)

            face_crop = frame[y1:y2, x1:x2]
            gender = predict_gender(face_crop)

            # Final behavior string
            behavior = sleep_status
            if phone_detected:
                behavior += " + Using Phone"

            # Draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{behavior} | {gender}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # ─── Display EAR and Frame ─────────
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img, channels="RGB")
    status_text.markdown(f"**🔍 Last Detected Status:** `{behavior} | {gender}`")

    time.sleep(0.2)

cap.release()


# --- head.py ---

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# ─── Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

frame_placeholder = st.empty()
status_text = st.markdown("**🔍 Last Detected Status:** None")

# ─── Load Models ───────────────────────────────────────────────────────
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 pretrained model (COCO)
gender_model = load_model("gender_classifier_model.h5")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ─── Constants ─────────────────────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1
CHIN = 152

EAR_THRESHOLD = 0.22
SLEEP_FRAMES = 40
HEAD_DOWN_THRESHOLD = 0.1  # Lower means stricter

closed_counter = 0
ear_queue = deque(maxlen=5)

# ─── Utility Functions ─────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def predict_gender(face_crop):
    try:
        if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
            return "Unknown"
        face = cv2.resize(face_crop, (128, 128))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        pred = gender_model.predict(face, verbose=0)[0][0]
        return "Female" if pred < 0.5 else "Male"
    except:
        return "Unknown"

# ─── Main Loop ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
ear = 0.3  # default EAR

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    sleep_status = "Awake"
    gender = "Unknown"
    phone_detected = False
    behavior = "Awake"

    # ─── Detect Phone with YOLO ───────
    yolo_results = yolo_model(frame, verbose=False)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_results.names[cls_id]
        if label.lower() == "cell phone":
            phone_detected = True
            break

    # ─── Detect Face & Sleep & Head ──────────
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            ear_queue.append(ear)
            smoothed_ear = np.mean(ear_queue)

            # Head bending
            nose_y = face_landmarks.landmark[NOSE_TIP].y
            chin_y = face_landmarks.landmark[CHIN].y
            head_bend_ratio = chin_y - nose_y

            # Decide status
            if smoothed_ear < EAR_THRESHOLD or head_bend_ratio > HEAD_DOWN_THRESHOLD:
                closed_counter += 1
                if closed_counter > SLEEP_FRAMES:
                    sleep_status = "Sleeping 😴"
            else:
                closed_counter = 0

            # Face Box
            x_coords = [p[0] for p in landmarks]
            y_coords = [p[1] for p in landmarks]
            x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
            x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)

            face_crop = frame[y1:y2, x1:x2]
            gender = predict_gender(face_crop)

            # Final behavior string
            behavior = sleep_status
            if phone_detected:
                behavior += " + Using Phone"

            # Draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{behavior} | {gender}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # ─── Display EAR and Frame ─────────
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img, channels="RGB")
    status_text.markdown(f"**🔍 Last Detected Status:** `{behavior} | {gender}`")

    time.sleep(0.2)

cap.release()


# --- loit.py ---

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load both models (one for chair, one fallback for person)
@st.cache_resource
def load_models():
    chair_model = YOLO(r"C:\Users\Lenovo\Desktop\FINAL\best.pt")
    fallback_model = YOLO("yolov8n.pt")
    return chair_model, fallback_model

chair_model, fallback_model = load_models()

# Streamlit interface
st.title("🪑 Real-Time Chair Occupancy Detection")
st.markdown("Detects if a chair is occupied using a YOLOv8 model.")

# Set up frame container
frame_display = st.empty()

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam

# Detection function
def detect_occupancy(frame):
    chair_result = chair_model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]
    chair_detected = chair_result.boxes and len(chair_result.boxes.cls) > 0

    if chair_detected:
        return frame, "Occupied ✅"

    # Fallback - detect person
    fallback_result = fallback_model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
    labels = fallback_result.boxes.cls.tolist() if fallback_result.boxes else []
    names = fallback_model.names
    person_detected = any(names[int(cls)] == "person" for cls in labels)

    if person_detected:
        return frame, "Occupied ✅"
    
    return frame, "Loitering ❌"

# Stream webcam with detections
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam input error.")
        break

    frame = cv2.flip(frame, 1)
    processed_frame, status = detect_occupancy(frame)

    # Add status text to frame
    cv2.putText(processed_frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Occupied" in status else (0, 0, 255), 2)

    # Display the result
    frame_display.image(processed_frame, channels="BGR")

cap.release()


# --- mouth.py ---

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from PIL import Image
import threading
import queue

# ─── Setup Streamlit ───────────────────────────────────────────────────
st.set_page_config(page_title="Employee Eating Detection", layout="wide")
st.title("🍽️ Real-time Detection: Employee Eating")

col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    status_text = st.markdown("**🔍 Last Detected Status:** None")

with st.sidebar:
    st.header("Settings")
    st.markdown("""
    **Eating Detection**:
    - Detects chewing via mouth movements using MediaPipe Face Mesh.
    - Ensure bright lighting, face 1-2 feet from webcam, no obstructions (e.g., glasses).
    - Test by opening/closing mouth 1-2 times over ~2s (like chewing food).
    - MAR should decrease (<0.15) when mouth opens, increase (>0.15) when closes.
    """)
    MOUTH_OPEN_THRESHOLD = st.slider("Mouth Open Threshold", 0.05, 0.5, 0.15, 0.05,
                                     help="Lower MAR indicates open mouth; adjust for sensitivity.")
    EATING_FRAMES = st.slider("Eating Frames Threshold", 5, 60, 8, 1,
                              help="Frames with mouth movement to detect eating (lower = faster).")

# ─── Load Models ───────────────────────────────────────────────────────
try:
    with st.spinner("Loading MediaPipe Face Mesh..."):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
except Exception as e:
    st.error(f"Error loading MediaPipe: {e}")
    st.stop()

# ─── Constants ─────────────────────────────────────────────────────────
MOUTH_OUTER = [13, 14, 78, 308]  # Top (13), bottom (14), left (78), right (308) mouth landmarks
eating_counter = 0
mouth_state_queue = deque(maxlen=10)  # Track mouth open/closed states
FRAME_SKIP = 2
TARGET_RESOLUTION = (640, 480)

# ─── Video Capture Thread ──────────────────────────────────────────────
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
            frame = cv2.resize(frame, TARGET_RESOLUTION)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    finally:
        cap.release()

thread = threading.Thread(target=video_capture, daemon=True)
thread.start()

# ─── Utility Functions ─────────────────────────────────────────────────
def mouth_aspect_ratio(landmarks, mouth_indices):
    """Calculate MAR to measure mouth openness for eating detection."""
    try:
        points = np.array([landmarks[i] for i in mouth_indices])
        vertical = np.linalg.norm(points[0] - points[1])  # Top to bottom
        horizontal = np.linalg.norm(points[2] - points[3])  # Left to right
        return vertical / horizontal
    except:
        return 0.0

# ─── Main Loop ─────────────────────────────────────────────────────────
mar = 0.0
frame_count = 0
last_ui_update = 0
last_warning_time = 0
try:
    while not stop_thread.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eating_status = ""
        behavior = "Not Eating"

        # ─── Detect Eating with MediaPipe Face Mesh ───────
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
                # Eating detection
                mar = mouth_aspect_ratio(landmarks, MOUTH_OUTER)
                mouth_state_queue.append(0 if mar > MOUTH_OPEN_THRESHOLD else 1)  # 0 for closed, 1 for open
                if len(mouth_state_queue) >= 10:
                    transitions = sum(1 for i in range(len(mouth_state_queue)-1) if mouth_state_queue[i] != mouth_state_queue[i+1])
                    if transitions >= 1:  # 0.5 open-close cycle
                        eating_counter += 1
                        if eating_counter > EATING_FRAMES:
                            eating_status = "Eating"
                    else:
                        eating_counter = 0

                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
                x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)

                behavior = eating_status if eating_status else "Not Eating"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{behavior}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        else:
            status_text.markdown("**🔍 Last Detected Status:** No face detected (check lighting, face 1-2 feet from webcam)")

        # ─── Display MAR and Frame ─────────
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_time = time.time()
        if current_time - last_ui_update >= 0.15:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
            frame_placeholder.image(img_pil, channels="RGB")
            status_text.markdown(f"**🔍 Last Detected Status:** {behavior}")
            last_ui_update = current_time

        # ─── Warn if Eating Not Detected After 30s ─────────
        if current_time - last_warning_time >= 30 and behavior == "Not Eating":
            st.warning(f"Eating not detected. MAR: {mar:.2f}. When mouth opens, MAR should decrease (<0.15); when closes, MAR should increase (>0.15). Chew 1-2 times over ~2s. Adjust lighting, face webcam directly, or lower MOUTH_OPEN_THRESHOLD to 0.05.")
            last_warning_time = current_time

finally:
    stop_thread.set()
    thread.join()
    face_mesh.close()