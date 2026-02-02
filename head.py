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
