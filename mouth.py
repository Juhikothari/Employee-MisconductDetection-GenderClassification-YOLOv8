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