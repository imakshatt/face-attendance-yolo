import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import argparse
import os

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["camera", "wsl"], required=True)
parser.add_argument("--input_video", type=str, help="Input video path for WSL mode")
parser.add_argument("--output_video", type=str, default="output.mp4")
args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
EMBEDDINGS_FILE = "embeddings/student_embeddings.pkl"
ATTENDANCE_FILE = "attendance/attendance.csv"

# -----------------------------
# Load embeddings
# -----------------------------
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_embeddings = np.array(known_embeddings).astype("float32")

known_roll_numbers = data["roll_numbers"]

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Video Source
# -----------------------------
if args.mode == "camera":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # safer on Windows
    output_writer = None

elif args.mode == "wsl":
    if args.input_video is None:
        raise ValueError("Provide --input_video in WSL mode")

    cap = cv2.VideoCapture(args.input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

# -----------------------------
# Attendance memory
# -----------------------------
marked_students = set()

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Roll_Number", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(roll):

    if roll in marked_students:
        return

    time_now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)

    df = pd.concat([
        df,
        pd.DataFrame([{
            "Roll_Number": roll,
            "Time": time_now
        }])
    ])

    df.to_csv(ATTENDANCE_FILE, index=False)

    marked_students.add(roll)

    print("Attendance Marked:", roll)

# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        continue

    # Ensure frame is uint8
    # frame = cv2.convertScaleAbs(frame)

    # # Resize frame for faster recognition
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # # Convert BGR -> RGB
    # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # # Ensure contiguous memory
    # rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
    # Convert frame to numpy array safely
    frame = np.array(frame)

    # Force uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Ensure 3 channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR → RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Ensure contiguous memory
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

    print("DEBUG FRAME:", rgb_small_frame.dtype, rgb_small_frame.shape)


    # -----------------------------
    # FACE RECOGNITION
    # -----------------------------
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    except Exception as e:
        print("Face recognition skipped:", e)
        continue

    # Loop through detected faces
    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):

        distances = face_recognition.face_distance(known_embeddings, encoding)

        idx = np.argmin(distances)
        min_dist = distances[idx]

        if min_dist < 0.5:
            roll = known_roll_numbers[idx]
            label = f"Roll:{roll}"
            mark_attendance(roll)

        else:
            label = "Unknown"

        # Scale coordinates back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # YOLO Detection
    # -----------------------------
    #results = model(frame)
    results = model(frame.copy())


    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for x1, y1, x2, y2 in boxes:

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -----------------------------
    # Display / Output
    # -----------------------------
    if args.mode == "camera":

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elif args.mode == "wsl":

        output_writer.write(frame)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()

if output_writer is not None:
    output_writer.release()

cv2.destroyAllWindows()
