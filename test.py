import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic.Holistic()
cap = cv2.VideoCapture(0)

def extract_keypoints(results):
    # สกัดจุดสำคัญของท่าทางร่างกาย
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    # สกัดจุดสำคัญของมือซ้าย (จากมุมมองกล้อง)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # สกัดจุดสำคัญของมือขวา (จากมุมมองกล้อง)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # รวมจุดสำคัญทั้งหมดเข้าด้วยกัน
    return np.concatenate([pose, lh, rh])

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_holistic.process(img)
    keypoints = extract_keypoints(results)
    print(keypoints)
    key = cv2.waitKey(1)
    cv2.imshow("pic",img)
    if key == ord("q"):
        break