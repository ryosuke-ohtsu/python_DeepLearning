import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def put_text_jp(img, text, org, font_scale=0.7, color=(255,255,255), thickness=2):
    if text is None or text == "":
        return

    if all(ord(ch) < 128 for ch in text):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    for candidate in [
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msmincho.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/ipaexg.ttf"
    ]:
        try:
            font_size = max(12, int(font_scale * 30))
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            font = None

    if font is None:
        font = ImageFont.load_default()

    draw.text(org, text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    img[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


TRAINER_DIR = "トレーナー"
if not os.path.exists(TRAINER_DIR):
    TRAINER_DIR = "trainer"
MODEL_PATH = os.path.join(TRAINER_DIR, "face_trainer.yml")
LABEL_PATH = os.path.join(TRAINER_DIR, "labels.txt")

if not os.path.exists(MODEL_PATH):
    print("学習データがありません。先に train_faces.py を実行してください。")
    exit()

if not os.path.exists(LABEL_PATH):
    print("labels.txt がありません。")
    exit()

id_to_name = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(",", 1)
        id_to_name[int(idx)] = name

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

print("q で終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = recognizer.predict(face_roi)

        # confidenceは小さいほど一致度が高い
        if confidence < 70:
            name = id_to_name.get(label_id, "Unknown")
            text = f"{name} ({confidence:.1f})"
        else:
            name = "Unknown"
            text = f"Unknown ({confidence:.1f})"

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        put_text_jp(frame, text, (x, y-30), font_scale=0.8, color=color, thickness=2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()