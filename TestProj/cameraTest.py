import cv2

# 顔検出用の学習済みデータ
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Webカメラを開く（通常は0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webカメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 顔に枠を描く
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顔の数を表示
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    # 画面表示
    cv2.imshow("Face Detection", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()