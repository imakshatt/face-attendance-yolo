# Face Attendance System using YOLO

This project implements a **Face Recognition Based Attendance System** using:

* **YOLOv8** for face detection
* **Face Recognition (dlib / face_recognition)** for identity matching
* **Python + OpenCV**
* **CSV file for attendance storage**

The system detects a student's face from a **camera or video file**, identifies the student, and records attendance automatically.

---

# System Architecture

```
Camera / Video
      │
      ▼
YOLOv8 Face Detection
      │
      ▼
Face Embedding Extraction
      │
      ▼
Student Database Matching
      │
      ▼
Attendance Marked in CSV
```

---

# Project Structure

```
face-attendance-yolo
│
├── dataset/
│   ├── 101/
│   ├── 102/
│   └── ...
│
├── attendance/
│   └── attendance.csv
│
├── embeddings/
│   └── student_embeddings.pkl
│
├── models/
│
├── outputs/
│
├── scripts/
│   ├── arrange_db.py
│   ├── create_embeddings.py
│   └── attendance_system.py
│
├── requirements.txt
├── setup.sh
└── README.md
```

---

# Requirements

Tested with:

```
Python 3.10.20
Ubuntu / WSL
```

---

# Quick Setup (Recommended)

Clone repository:

```
git clone https://github.com/imakshatt/face-attendance-yolo.git
cd face-attendance-yolo
```

Run setup script:

```
bash setup.sh
```

This script will automatically:

* install required system dependencies
* create Python virtual environment
* install all Python packages
* prepare required folders

---

# Activate Environment

Whenever you open a new terminal run:

```
source yolo_env/bin/activate
```

---

# Dataset Format

Dataset is organized by **Roll Number**.

Example:

```
dataset/
    101/
        img1.jpg
        img2.jpg

    102/
        img1.jpg
```

Each folder represents **one student**.

Recommended:

```
5–10 images per student
```

---

# Step 1 — Generate Face Embeddings

Convert student images into facial feature vectors.

```
cd scripts
python create_embeddings.py
```

This creates:

```
embeddings/student_embeddings.pkl
```

---

# Step 2 — Run Attendance System

### Camera Mode (Normal Linux)

```
python attendance_system.py --mode camera
```

This will open the webcam and detect students.

---

### Video Mode (WSL Compatible)

```
python attendance_system.py \
--mode wsl \
--input_video ../dataset/output.mp4 \
--output_video ../outputs/result.mp4
```

This mode:

* reads video file
* processes frames
* detects faces
* marks attendance
* saves processed video

---

# Attendance Output

Attendance is saved in:

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
* WSL video processing mode

---

# Troubleshooting

If `face_recognition` fails to install:

```
pip install dlib
pip install face_recognition
```

If OpenCV GUI does not work in WSL, use **video mode** instead.

---

# Future Improvements

Possible enhancements:

* YOLOv8-face detection model
* Face tracking for better stability
* SQLite database instead of CSV
* Real-time dashboard UI
* Multi-person detection

---

# Author & Developer

Akshat
GitHub: https://github.com/imakshatt
