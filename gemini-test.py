import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic # สำหรับท่าทางร่างกายและมือ
mp_drawing = mp.solutions.drawing_utils # สำหรับวาดจุดลงบนภาพ

def extract_keypoints(results):
    # สกัดจุดสำคัญของท่าทางร่างกาย
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    # สกัดจุดสำคัญของมือซ้าย (จากมุมมองกล้อง)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # สกัดจุดสำคัญของมือขวา (จากมุมมองกล้อง)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # รวมจุดสำคัญทั้งหมดเข้าด้วยกัน
    return np.concatenate([pose, lh, rh])

# Path ของวิดีโอ
DATA_PATH = os.path.join('data') 
actions = np.array(['hello', 'thankyou']) # คำศัพท์ของเรา

# จำนวนเฟรมที่จะใช้ในแต่ละท่าทาง (ลำดับเวลา)
no_sequences = 30 # จำนวนตัวอย่างวิดีโอสำหรับแต่ละท่าทาง
sequence_length = 30 # จำนวนเฟรมในแต่ละลำดับ (30 เฟรม = 1 วินาที ที่ 30 fps)

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# วนลูปเพื่อประมวลผลวิดีโอ
for action in actions:
    print(1)
    for sequence in range(no_sequences):
        print(2)
        video_path = os.path.join(DATA_PATH, action, f'{action}_{sequence+1:03d}.mp4') # สมมติชื่อไฟล์ตามตัวอย่างด้านบน

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(video_path)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            frame_num = 0
            while cap.isOpened() and frame_num < sequence_length: # จำกัดจำนวนเฟรมที่ประมวลผลต่อวิดีโอ
                ret, frame = cap.read()
                if not ret:
                    break
                
                # ทำการตรวจจับจุดสำคัญ
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                # สกัดและบันทึกจุดสำคัญ
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                frame_num += 1
                
            cap.release()

print("Keypoint extraction complete!")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# สร้าง array สำหรับเก็บข้อมูลทั้งหมด
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(actions.index(action)) # เปลี่ยนชื่อคำศัพท์เป็นตัวเลข (0 สำหรับ hello, 1 สำหรับ thankyou)

X = np.array(sequences) # Input data (ลำดับของ keypoints)
y = to_categorical(labels).astype(int) # Output data (one-hot encoded labels)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) # 15% สำหรับ test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42) # 15% ของที่เหลือ (ประมาณ 15% ของข้อมูลทั้งหมด)

print(f"X_train shape: {X_train.shape}") # (จำนวนตัวอย่าง, sequence_length, จำนวน keypoints)
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# จำนวนคลาสของเรา (hello, thankyou)
num_classes = len(actions) 
# ขนาดของ Input (sequence_length, จำนวน keypoints)
input_shape = (sequence_length, X.shape[2])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
model.add(Dropout(0.2)) # เพิ่ม Dropout เพื่อป้องกัน Overfitting
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu')) # return_sequences=False สำหรับ layer สุดท้ายก่อน Dense
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) # Output layer: ใช้ softmax สำหรับการจำแนกหลายคลาส

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

# เริ่มการฝึก
history = model.fit(
    X_train, y_train, 
    epochs=200, # จำนวน Epochs
    batch_size=32, # Batch Size
    validation_data=(X_val, y_val), 
    callbacks=[tb_callback]
)

# บันทึกโมเดลหลังจากฝึกเสร็จ
model.save('sign_language_model.h5')
print("Model saved as sign_language_model.h5")