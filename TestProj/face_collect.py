import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 設定
# -----------------------------
DATASET_DIR = "データセット"


def put_text_jp(img, text, org, font_scale=0.7, color=(255,255,255), thickness=2):
    """日本語を含むテキストを OpenCV に表示（PIL経由）"""
    if text is None or text == "":
        return

    # 英数字のみなら cv2 で描画
    if all(ord(ch) < 128 for ch in text):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return

    # PIL に変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # フォント指定（Windowsの日本語フォント）
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

    # OpenCV座標系とPILは左上基準で同じ
    draw.text(org, text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    img[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 既存の dataset フォルダがある場合はそちらを使う
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


def normalize_person_name(name):
    # Windows禁止文字と余分な空白を置き換えて安全なフォルダ名に
    name = name.strip()
    if not name:
        return ""
    for bad in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "\n", "\r"]:
        name = name.replace(bad, "_")
    # 全角スペースも半角スペースも置換
    name = name.replace(" ", "_").replace("　", "_")
    return name

name = input("登録する名前を入力してください: ").strip()
name = normalize_person_name(name)

if not name:
    print("名前が空です。終了します。")
    exit()

person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def imwrite_unicode(path, image):
    """日本語パス対応で画像保存"""
    try:
        success = cv2.imwrite(path, image)
        if success:
            return True
    except Exception:
        success = False

    # cv2.imwriteで失敗した場合、imencode経由で保存
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    ret, buf = cv2.imencode(ext, image)
    if not ret:
        return False
    try:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


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
        put_text_jp(frame, name, (x, y-30), font_scale=0.8, color=(0, 255, 0), thickness=2)

    put_text_jp(frame, f"Name: {name}", (10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2)
    put_text_jp(frame, f"Saved: {count}", (10, 60), font_scale=0.8, color=(255, 255, 255), thickness=2)
    put_text_jp(frame, "Press S to save / Q to quit", (10, 90), font_scale=0.7, color=(255, 255, 0), thickness=2)

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
        if imwrite_unicode(filename, face_img):
            print(f"保存: {filename}")
        else:
            print(f"保存失敗: {filename}")
            count -= 1


cap.release()
cv2.destroyAllWindows()