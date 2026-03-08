# Face Attendance System using YOLO

This project implements a **Face Recognition Based Attendance System** using:

* **YOLOv8** for face detection
* **Face Recognition (dlib / FaceNet)** for identity matching
* **Python + OpenCV**
* **CSV file for attendance storage**

The system detects a student's face from a camera or video file, identifies the student, and records attendance automatically.

---

# Project Structure

```
face-attendance-yolo/
│
├── attendance/
│   └── attendance.csv
│
├── dataset/
│   ├── 101/
│   ├── 102/
│   └── ...
│
├── embeddings/
│   └── student_embeddings.pkl
│
├── models/
│   └── yolov8n.pt
│
├── outputs/
│   └── result.mp4
│
├── scripts/
│   ├── arrange_db.py
│   ├── create_embeddings.py
│   └── attendance_system.py
│
├── requirements.txt
└── README.md
```

---

# Requirements

This project was tested with:

```
Python 3.10.20
Ubuntu / WSL
```

---

# Step 1 — Install System Dependencies

This step prevents **dlib / face_recognition build errors**.

Run:

```bash
sudo apt update

sudo apt install -y \
cmake \
build-essential \
libopenblas-dev \
liblapack-dev \
libx11-dev \
libgtk-3-dev \
python3-dev
```

---

# Step 2 — Create Virtual Environment

Navigate to the project directory:

```bash
python3 -m venv yolo_env
```

Activate environment:

```bash
source yolo_env/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# Step 3 — Install Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

This will install:

* ultralytics
* opencv
* numpy
* pandas
* face_recognition
* scikit-learn

---

# Step 4 — Prepare Dataset

Dataset should be organized by **Roll Number**.

Example:

```
dataset/
    101/
        img1.jpg
        img2.jpg
    102/
        img1.jpg
```

Each student folder contains multiple face images.

Recommended: **5–10 images per student**

---

# Step 5 — Generate Face Embeddings

Run:

```bash
cd scripts
python create_embeddings.py
```

This will generate:

```
embeddings/student_embeddings.pkl
```

This file contains encoded facial features of all students.

---

# Step 6 — Run Attendance System

## Camera Mode (Normal Linux)

```
python attendance_system.py --mode camera
```

---

## Video Mode (WSL Compatible)

```
python attendance_system.py \
--mode wsl \
--input_video input.mp4 \
--output_video output.mp4
```

This will:

1. Process the video
2. Detect faces
3. Identify students
4. Mark attendance
5. Save processed video

Output:

```
outputs/result.mp4
attendance/attendance.csv
```

---

# Attendance File

Attendance is stored in:

```
attendance/attendance.csv
```

Example:

```
Roll_Number,Time
101,10:31:11
102,10:32:04
```

---

# Features

* YOLOv8 based detection
* Face recognition using embeddings
* CSV attendance logging
* Duplicate attendance prevention
* Camera mode
* WSL video mode

---

# Notes

If `face_recognition` fails to install:

```
pip install dlib
pip install face_recognition
```

---

# Future Improvements

Possible improvements:

* YOLOv8-face model
* Multi-face tracking
* GUI dashboard
* SQLite attendance database
* Real-time analytics
