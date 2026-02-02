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
import os
from datetime import datetime
import pandas as pd

# ─── PATH / CONFIG ─────────────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Users\Juhi\Downloads\FINAL\yolov8n.pt"
CHAIR_MODEL_PATH = r"C:\Users\Juhi\Downloads\FINAL\best.pt"
GENDER_MODEL_PATH = r"C:\Users\Juhi\Downloads\FINAL\gender_classifier_model.h5"
LOG_FILE_PATH = r"C:\Users\Juhi\Downloads\FINAL\misconduct_log.csv"  # ensure parent exists

os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# ─── Logging Buffer ────────────────────────────────────────────────────
log_buffer = []
log_lock = threading.Lock()

def log_event(event_type, details):
    with log_lock:
        log_buffer.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Event": event_type,
            "Details": details
        })

def flush_logs():
    with log_lock:
        if not log_buffer:
            return
        df = pd.DataFrame(log_buffer)
        if os.path.exists(LOG_FILE_PATH):
            df.to_csv(LOG_FILE_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE_PATH, index=False)
        log_buffer.clear()

def start_log_flusher(interval=10):
    def loop():
        while True:
            time.sleep(interval)
            flush_logs()
    t = threading.Thread(target=loop, daemon=True)
    t.start()

start_log_flusher()

# ─── Streamlit UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="Employee Misconduct Detection", layout="wide")
st.title("🧑‍💼 Real-time Detection: Employee Misconduct")

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
    """)
    MOUTH_OPEN_THRESHOLD = st.slider("Mouth Open Threshold", 0.05, 0.5, 0.15, 0.05,
                                     help="Lower MAR for open mouth; adjust for sensitivity.")
    EATING_FRAMES = st.slider("Eating Frames Threshold", 5, 60, 8, 1,
                              help="Frames with mouth movement to detect eating.")
    st.markdown("""
    **Sleeping & Head Bending Detection**:
    - Detects closed eyes (EAR) or downward head tilt.
    """)
    EYE_CLOSURE_THRESHOLD = st.slider("Eye Closure Threshold (EAR)", 0.1, 0.5, 0.22, 0.01,
                                      help="Lower for stricter eye closure detection.")
    SLEEP_FRAMES = st.slider("Sleep Frames Threshold", 20, 60, 40, 1,
                             help="Frames with closed eyes or head tilt to detect sleeping.")
    HEAD_DOWN_THRESHOLD = st.slider("Head Down Threshold", 0.05, 0.5, 0.1, 0.05,
                                    help="Higher for stricter head bending detection.")
    st.markdown("""
    **Phone & Crowd Detection**:
    - Detects phone usage and crowding using YOLOv8.
    """)
    CROWD_THRESHOLD = st.slider("Crowd Size Threshold", 2, 10, 3, 1,
                                help="Number of people to flag as crowding.")
    st.markdown("""
    **Chair Occupancy**:
    - Detects occupied chairs or people using YOLOv8.
    """)
    st.markdown("""
    **Gender Detection**:
    - Predicts gender (0 = Female, 1 = Male, threshold 0.5).
    - Shows confidence score (0 to 1).
    """)

# ─── Load Models ───────────────────────────────────────────────────────
try:
    with st.spinner("Loading models..."):
        if not os.path.exists(YOLO_MODEL_PATH):
            st.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
            st.stop()
        if not os.path.exists(CHAIR_MODEL_PATH):
            st.warning(f"Chair model not found at {CHAIR_MODEL_PATH}, fallback to person detection only.")
        if not os.path.exists(GENDER_MODEL_PATH):
            st.error(f"Gender model not found at {GENDER_MODEL_PATH}")
            st.stop()

        yolo_model = YOLO(YOLO_MODEL_PATH)
        try:
            chair_model = YOLO(CHAIR_MODEL_PATH)
        except Exception:
            chair_model = None
        gender_model = load_model(GENDER_MODEL_PATH)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ─── Constants & State ─────────────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_OUTER = [13, 14, 78, 308]
NOSE_TIP = 1
CHIN = 152
FRAME_SKIP = 2
TARGET_RESOLUTION = (640, 480)

closed_counter = 0
eating_counter = 0
ear_queue = deque(maxlen=5)
mouth_state_queue = deque(maxlen=10)
prev_face_crop = None
prev_gender = "Unknown"
prev_confidence = 0.0

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

# ─── Helper Functions ─────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices):
    try:
        pts = np.array([landmarks[i] for i in eye_indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C + 1e-8)
    except:
        return 0.3

def mouth_aspect_ratio(landmarks, mouth_indices):
    try:
        pts = np.array([landmarks[i] for i in mouth_indices])
        vertical = np.linalg.norm(pts[0] - pts[1])
        horizontal = np.linalg.norm(pts[2] - pts[3])
        return vertical / (horizontal + 1e-8)
    except:
        return 0.0

def predict_gender(face_crop, prev_face=None):
    if face_crop.size == 0 or face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
        return None, 0.0
    if prev_face is not None and face_crop.shape == prev_face.shape:
        diff = np.mean(np.abs(face_crop.astype(float) - prev_face.astype(float)))
        if diff < 15:
            return None, 0.0  # skip if almost same
    try:
        face = cv2.resize(face_crop, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        pred = gender_model.predict(face, verbose=0)[0][0]
        gender = "Female" if pred < 0.5 else "Male"
        return gender, pred
    except:
        return None, 0.0

def detect_occupancy(frame):
    if chair_model is not None:
        try:
            chair_result = chair_model(frame, verbose=False)[0]
            if len(chair_result.boxes) > 0:
                return "Occupied ✅"
        except:
            pass
    # fallback to person detection
    try:
        fallback_result = yolo_model(frame, verbose=False)[0]
        for box in fallback_result.boxes:
            cls_id = int(box.cls[0])
            label = fallback_result.names.get(cls_id, "").lower()
            if label == "person":
                return "Occupied ✅"
    except:
        pass
    return "Loitering ❌"

# ─── Main Loop ─────────────────────────────────────────────────────────
ear = 0.3
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
        sleep_status = "Awake"
        eating_status = ""
        phone_detected = False
        crowd_detected = False
        chair_status = "Loitering ❌"
        gender = prev_gender
        confidence = prev_confidence
        behavior = "Awake"

        # YOLO: phone + crowd
        if frame_count % (FRAME_SKIP * 2) == 0:
            try:
                yolo_results = yolo_model(frame, verbose=False)[0]
                person_count = 0
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_results.names.get(cls_id, "").lower()
                    conf = box.conf.item()
                    if conf < 0.5:
                        continue
                    if label == "person":
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    elif "phone" in label:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.putText(frame, "Phone", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if person_count >= CROWD_THRESHOLD:
                    crowd_detected = True
            except:
                pass

        # Chair occupancy
        chair_status = detect_occupancy(frame)

        # Face mesh & behaviors
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

                # Eating detection (MAR transitions)
                mar = mouth_aspect_ratio(landmarks, MOUTH_OUTER)
                mouth_state_queue.append(0 if mar > MOUTH_OPEN_THRESHOLD else 1)
                if len(mouth_state_queue) >= 10:
                    transitions = sum(1 for i in range(len(mouth_state_queue)-1)
                                      if mouth_state_queue[i] != mouth_state_queue[i+1])
                    if transitions >= 1:
                        eating_counter += 1
                        if eating_counter > EATING_FRAMES:
                            eating_status = "Eating"
                            log_event("Eating", f"Detected eating behavior (MAR={mar:.2f})")
                    else:
                        eating_counter = 0

                # Sleep detection (EAR + head bend)
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                ear_queue.append(ear)
                smoothed_ear = np.mean(ear_queue)

                nose_y = face_landmarks.landmark[NOSE_TIP].y
                chin_y = face_landmarks.landmark[CHIN].y
                head_bend_ratio = chin_y - nose_y

                if smoothed_ear < EYE_CLOSURE_THRESHOLD or head_bend_ratio > HEAD_DOWN_THRESHOLD:
                    closed_counter += 1
                    if closed_counter > SLEEP_FRAMES:
                        sleep_status = "Sleeping 😴"
                        log_event("Sleep", "Detected sleeping/head-down")
                else:
                    closed_counter = 0

                # Gender
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
                x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)
                face_crop = frame[y1:y2, x1:x2]
                gender_pred, confidence_pred = predict_gender(face_crop, prev_face_crop)
                if gender_pred is not None:
                    gender = gender_pred
                    confidence = confidence_pred
                    prev_gender = gender
                    prev_confidence = confidence
                    prev_face_crop = face_crop.copy()

                # Compose behavior string
                behavior_components = [sleep_status]
                if eating_status:
                    behavior_components.append("Eating")
                if phone_detected:
                    behavior_components.append("Using Phone")
                if crowd_detected:
                    behavior_components.append("Crowding")
                behavior = " + ".join(filter(None, behavior_components))

                # Draw face box & overlay
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{behavior} | {gender} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            if crowd_detected:
                behavior = "Crowding"

        # Overlay metrics
        cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Chair: {chair_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # UI update throttle
        current_time = time.time()
        if current_time - last_ui_update >= 0.15:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
            frame_placeholder.image(img_pil, channels="RGB")
            status_text.markdown(f"**🔍 Last Detected Status:** {behavior} | Chair: {chair_status} | {gender} (Confidence: {confidence:.2f})")
            last_ui_update = current_time

        # Warning if no eating detected after 30s
        if current_time - last_warning_time >= 30 and "Eating" not in behavior:
            st.warning(f"Eating not detected. MAR: {mar:.2f}. Chew 1-2 times over ~2s; adjust threshold or lighting.") 
            last_warning_time = current_time

finally:
    stop_thread.set()
    thread.join()
    face_mesh.close()
    flush_logs()
