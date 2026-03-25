"""
顔認識システム メイン実行スクリプト
複数人の登録から学習、認識までを統一管理
"""
import os
import sys
import cv2
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


def imwrite_unicode(path, image):
    try:
        success = cv2.imwrite(path, image)
        if success:
            return True
    except Exception:
        success = False

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


DATASET_DIR = "データセット"
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = "dataset"

TRAINER_DIR = "トレーナー"
if not os.path.exists(TRAINER_DIR):
    TRAINER_DIR = "trainer"


def ensure_dirs():
    """必要なディレクトリ作成"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TRAINER_DIR, exist_ok=True)


def show_main_menu():
    """メインメニュー表示"""
    print("\n" + "=" * 50)
    print("顔認識システム")
    print("=" * 50)
    print("1. 顔画像の撮影・登録")
    print("2. モデルの学習")
    print("3. 顔の認識")
    print("4. 登録済み人物の確認")
    print("5. 終了")
    print("=" * 50)
    choice = input("選択してください (1-5): ").strip()
    return choice


def collect_faces():
    """複数人の顔画像を撮影・保存"""
    print("\n【顔画像の撮影・登録】")
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    while True:
        name = input("\n登録する人の名前を入力 (終了: 空欄で入力): ").strip()
        
        if not name:
            print("登録を終了します。")
            break
        
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # 既に保存されている枚数をカウント
        existing_count = len([f for f in os.listdir(person_dir) 
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        
        print(f"\n[{name}の撮影を開始します]")
        print(f"既に保存済み: {existing_count}枚")
        print("操作:")
        print("  S キー: 顔画像を保存")
        print("  Q キー: この人物の登録を終了")
        print("  このメニューに戻る場合は Q → 空欄入力")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ カメラを開けませんでした。")
            continue
        
        count = existing_count
        
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
            
            # 顔を矩形で囲む
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                put_text_jp(frame, name, (x, y-30), font_scale=0.8, color=(0, 255, 0), thickness=2)
            
            # 情報表示
            put_text_jp(frame, f"Person: {name}", (10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2)
            put_text_jp(frame, f"Saved: {count}", (10, 60), font_scale=0.8, color=(255, 255, 255), thickness=2)
            put_text_jp(frame, "S:Save / Q:Next Person", (10, 90), font_scale=0.7, color=(255, 255, 0), thickness=2)
            
            cv2.imshow(f"Face Collection - {name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q") or key == ord("Q"):
                break
            
            if key == ord("s") or key == ord("S"):
                if len(faces) == 0:
                    print("⚠️  顔が見つかりません")
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
                    print(f"✓ 保存: {filename}")
                else:
                    print(f"保存失敗: {filename}")
                    count -= 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count > existing_count:
            print(f"✓ {name}: {count - existing_count}枚を新規保存しました")
        else:
            print(f"⚠️  {name}: 新規保存はありません")


def train_model():
    """保存した顔画像でモデルを学習"""
    print("\n【モデルの学習】")
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    label_ids = {}
    current_id = 0
    
    x_train = []
    y_labels = []
    
    # データセットから学習用データを準備
    person_count = 0
    image_count = 0
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1
        
        label_id = label_ids[person_name]
        person_images = 0
        
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            path = os.path.join(person_dir, file_name)
            
            try:
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
                
                person_images += 1
                image_count += 1
            except Exception as e:
                print(f"⚠️  {path} の読み込みに失敗: {e}")
        
        if person_images > 0:
            print(f"  {person_name}: {person_images}枚")
            person_count += 1
    
    if len(x_train) == 0:
        print("❌ 学習用画像が見つかりませんでした。")
        print(f"   {DATASET_DIR} フォルダに顔画像を保存してください。")
        return
    
    print(f"\n学習中...")
    print(f"  登録人数: {person_count}人")
    print(f"  学習用画像: {image_count}枚")
    
    recognizer.train(x_train, np.array(y_labels))
    
    model_path = os.path.join(TRAINER_DIR, "face_trainer.yml")
    recognizer.save(model_path)
    
    label_path = os.path.join(TRAINER_DIR, "labels.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        for name, idx in sorted(label_ids.items(), key=lambda x: x[1]):
            f.write(f"{idx},{name}\n")
    
    print(f"\n✓ 学習完了！")
    print(f"  モデル: {model_path}")
    print(f"  ラベル: {label_path}")


def recognize():
    """カメラで顔認識を実行"""
    print("\n【顔の認識】")
    
    model_path = os.path.join(TRAINER_DIR, "face_trainer.yml")
    label_path = os.path.join(TRAINER_DIR, "labels.txt")
    
    if not os.path.exists(model_path):
        print("❌ 学習データがありません。")
        print("   先に『2. モデルの学習』を実行してください。")
        return
    
    if not os.path.exists(label_path):
        print("❌ labels.txt がありません。")
        return
    
    # ラベルを読み込み
    id_to_name = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(",", 1)
            id_to_name[int(idx)] = name
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ カメラを開けませんでした。")
        return
    
    print("顔認識を開始します。Q キーで終了します。")
    
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
            if confidence < 60:
                name = id_to_name.get(label_id, "Unknown")
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)
            elif confidence < 100:
                name = id_to_name.get(label_id, "Unknown")
                text = f"{name}? ({confidence:.1f})"
                color = (0, 165, 255)
            else:
                name = "Unknown"
                text = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            put_text_jp(frame, text, (x, y-30), font_scale=0.8, color=color, thickness=2)
        
        put_text_jp(frame, "Q:Quit", (10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2)
        
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def show_registered_people():
    """登録済み人物を表示"""
    print("\n【登録済み人物の確認】")
    
    people_data = {}
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        image_count = len([f for f in os.listdir(person_dir) 
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        
        if image_count > 0:
            people_data[person_name] = image_count
    
    if not people_data:
        print("登録済み人物はいません。")
        return
    
    print("\n登録済み人物:")
    print("-" * 40)
    total = 0
    for name, count in sorted(people_data.items()):
        print(f"  {name}: {count}枚")
        total += count
    
    print("-" * 40)
    print(f"合計: {len(people_data)}人, {total}枚")


def main():
    """メインループ"""
    ensure_dirs()
    
    while True:
        choice = show_main_menu()
        
        if choice == "1":
            collect_faces()
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize()
        elif choice == "4":
            show_registered_people()
        elif choice == "5":
            print("終了します。")
            break
        else:
            print("❌ 無効な選択です。")
        
        input("\nEnter キーを押してメインメニューに戻ります...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)
