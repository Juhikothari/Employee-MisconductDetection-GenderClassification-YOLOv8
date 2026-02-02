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
