import os
import face_recognition
import pickle
from tqdm import tqdm

DATASET_DIR = "../dataset"
EMBEDDINGS_FILE = "../embeddings/student_embeddings.pkl"

known_embeddings = []
known_roll_numbers = []

for roll in os.listdir(DATASET_DIR):

    student_path = os.path.join(DATASET_DIR, roll)

    if not os.path.isdir(student_path):
        continue

    for img_name in os.listdir(student_path):

        img_path = os.path.join(student_path, img_name)

        image = face_recognition.load_image_file(img_path)

        faces = face_recognition.face_encodings(image)

        if len(faces) > 0:
            embedding = faces[0]

            known_embeddings.append(embedding)
            known_roll_numbers.append(roll)

print("Total embeddings:", len(known_embeddings))

data = {
    "embeddings": known_embeddings,
    "roll_numbers": known_roll_numbers
}

with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved to", EMBEDDINGS_FILE)