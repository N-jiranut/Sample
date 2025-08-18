import cv2, time, joblib, math
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]
# pca = joblib.load("pca.pkl")
# # pca = PCA(n_components=100)

name = "M8-17-2025-moving_hands"

model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()
    
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

def main_save():
    LH = [0 for _ in range(42)]
    RH = [0 for _ in range(42)]
    BO = [0 for _ in range(14)]
    row=[]
    condition=0
    con = 0
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    perframe = math.floor(allframe/15)
    for woah in range(allframe):
        ret, frame = cap.read()
        if not ret: 
            break
        if condition%perframe == 0 and con < 15:
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            hand_result = hands.process(frame)
            pose_results = pose.process(frame)
            if hand_result.multi_hand_landmarks:
                RH=[]
                LH=[]
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    for lm in hand_landmarks.landmark: 
                        x, y= lm.x, lm.y         
                        if handedness == "Left" and len(LH)<42:
                            LH.extend([x,y])
                        if handedness == "Right" and len(RH)<42:
                            RH.extend([x,y])

                if len(LH) <= 0:
                    LH = [0 for _ in range(42)]
                row.extend(LH)
                if len(RH) <= 0:
                    RH = [0 for _ in range(42)]
                row.extend(RH)
                
                if pose_results.pose_landmarks:
                    BO=[]
                    landmarks = pose_results.pose_landmarks.landmark
                    for id,lm in enumerate(landmarks):
                        x, y = lm.x, lm.y
                        if id in pose_take:
                            cv2.circle(frame ,(round(x*720),round(y*480)), 1, (0,0,255), 7)
                            BO.extend([x, y])    
                else:
                    BO.extend([0 for _ in range(14)])
                row.extend(BO)
                con += 1
                 
            cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord('q'):
           break
        condition+=1
    
    if con < 15:
        foradd=[]
        for object in [LH,RH,BO]:
            foradd.extend(object)
        row.extend(foradd*(15-con))
    return row

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#     cv2.imshow("Video",frame)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         data = main_save()
#         data = np.array(data)
#         data = data.reshape(1, -1)
#         data = pca.transform(data)
#         # print(len(data))
#         # data.append(main_save())

#         landmarks_np = np.array(data).reshape(1, -1)
#         pred = model.predict(landmarks_np)
#         index = np.argmax(pred)
#         label = class_names[index]
#         print("Prediction:", label, "Confidence:", pred[0][index])

#     if key == ord("q"):
#         break

# cap = cv2.VideoCapture("./Main_video/tilted/sick/WIN_20250808_18_44_16_Pro.mp4")
cap = cv2.VideoCapture("C:/Users/User/OneDrive/Pictures/ม้วนฟิล์ม/WIN_20250817_12_08_11_Pro.mp4")

data = main_save()

landmarks_np = np.array(data).reshape(1, -1)
pred = model.predict(landmarks_np)
index = np.argmax(pred)
label = class_names[index]

print(pred[0][index])
print(label)