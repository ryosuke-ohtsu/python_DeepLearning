import cv2
import os
import numpy as np
from PIL import Image

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
os.makedirs(TRAINER_DIR, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

label_ids = {}
current_id = 0

x_train = []
y_labels = []

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    if person_name not in label_ids:
        label_ids[person_name] = current_id
        current_id += 1

    label_id = label_ids[person_name]

    for file_name in os.listdir(person_dir):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(person_dir, file_name)

        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image, "uint8")

        faces = detector.detectMultiScale(
            image_array,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            x_train.append(image_array)
            y_labels.append(label_id)
        else:
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                x_train.append(roi)
                y_labels.append(label_id)

if len(x_train) == 0:
    print("学習用画像が見つかりませんでした。")
    exit()

recognizer.train(x_train, np.array(y_labels))
recognizer.save(os.path.join(TRAINER_DIR, "face_trainer.yml"))

with open(os.path.join(TRAINER_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    for name, idx in label_ids.items():
        f.write(f"{idx},{name}\n")

print("学習完了")
print("保存先:")
print("  trainer/face_trainer.yml")
print("  trainer/labels.txt")