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

parser.add_argument("--mode",
                    choices=["camera","wsl"],
                    required=True)

parser.add_argument("--input_video",
                    type=str,
                    help="Input video path for WSL mode")

parser.add_argument("--output_video",
                    type=str,
                    default="output.mp4")

args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
EMBEDDINGS_FILE = "../embeddings/student_embeddings.pkl"
ATTENDANCE_FILE = "../attendance/attendance.csv"

# -----------------------------
# Load embeddings
# -----------------------------
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_roll_numbers = data["roll_numbers"]

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Video Source
# -----------------------------
if args.mode == "camera":

    cap = cv2.VideoCapture(0)
    output_writer = None

elif args.mode == "wsl":

    if args.input_video is None:
        raise ValueError("Provide --input_video in WSL mode")

    cap = cv2.VideoCapture(args.input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_writer = cv2.VideoWriter(
        args.output_video,
        fourcc,
        fps,
        (width, height)
    )

# -----------------------------
# Attendance memory
# -----------------------------
marked_students = set()

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Roll_Number","Time"])
    df.to_csv(ATTENDANCE_FILE,index=False)

def mark_attendance(roll):

    if roll in marked_students:
        return

    time_now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)

    new_row = {
        "Roll_Number": roll,
        "Time": time_now
    }

    df = pd.concat([df,pd.DataFrame([new_row])])

    df.to_csv(ATTENDANCE_FILE,index=False)

    marked_students.add(roll)

    print("Attendance Marked:",roll)

# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1,y1,x2,y2 = map(int,box)

            face = frame[y1:y2,x1:x2]

            if face.size == 0:
                continue

            rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb)

            if len(encodings) == 0:
                continue

            embedding = encodings[0]

            distances = face_recognition.face_distance(
                known_embeddings,
                embedding
            )

            idx = np.argmin(distances)
            min_dist = distances[idx]

            if min_dist < 0.5:

                roll = known_roll_numbers[idx]

                label = f"Roll:{roll}"

                mark_attendance(roll)

            else:
                label = "Unknown"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,
                        label,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

    # -----------------------------
    # Output behavior
    # -----------------------------
    if args.mode == "camera":

        cv2.imshow("Attendance",frame)

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