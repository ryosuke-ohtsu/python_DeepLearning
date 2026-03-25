import cv2
import os

# -----------------------------
# 設定
# -----------------------------
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

name = input("登録する名前を入力してください: ").strip()

if not name:
    print("名前が空です。終了します。")
    exit()

person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

count = len([f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")])

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

print("操作:")
print("  s : 顔画像を保存")
print("  q : 終了")

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Name: {name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Saved: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press S to save / Q to quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Collect", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        if len(faces) == 0:
            print("顔が見つかりません")
            continue

        x, y, w, h = faces[0]

        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_img = gray[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (200, 200))

        count += 1
        filename = os.path.join(person_dir, f"{name}_{count:03d}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"保存: {filename}")

cap.release()
cv2.destroyAllWindows()