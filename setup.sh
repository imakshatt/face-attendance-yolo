#!/bin/bash

set -e

echo "======================================"
echo " Face Attendance System Setup"
echo "======================================"

echo ""
echo "Step 1: Updating system packages..."
sudo apt update

echo ""
echo "Step 2: Installing system dependencies (required for dlib)..."

sudo apt install -y \
cmake \
build-essential \
libopenblas-dev \
liblapack-dev \
libx11-dev \
libgtk-3-dev \
python3-dev \
python3-pip

echo ""
echo "Step 3: Creating Python virtual environment..."

if [ ! -d "yolo_env" ]; then
    python3 -m venv yolo_env
fi

echo ""
echo "Step 4: Activating virtual environment..."

source yolo_env/bin/activate

echo ""
echo "Step 5: Upgrading pip..."

pip install --upgrade pip

echo ""
echo "Step 6: Installing Python dependencies..."

pip install -r requirements.txt

echo ""
echo "Step 7: Ensuring required directories exist..."

mkdir -p embeddings
mkdir -p attendance
mkdir -p outputs
mkdir -p models

echo ""
echo "======================================"
echo " Setup Complete!"
echo "======================================"

echo ""
echo "Activate environment with:"
echo "source yolo_env/bin/activate"

echo ""
echo "Next steps:"
echo "cd scripts"
echo "python create_embeddings.py"

echo ""
echo "Then run attendance system:"
echo "python attendance_system.py --mode camera"
echo ""
echo "Or for WSL video mode:"
echo "python attendance_system.py --mode wsl --input_video ../dataset/output.mp4"