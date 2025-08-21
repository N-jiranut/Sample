import cv2, time, joblib, math, os
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]
order = [4,9,14,19,24,29,34,39,44,49]
# pca = joblib.load("pca.pkl")
# # pca = PCA(n_components=100)

name = "M8-21-2025-moving_hands_full"

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

# cap = cv2.VideoCapture("./Main_video/tilted/why/WIN_20250808_17_32_08_Pro.mp4")
# cap = cv2.VideoCapture("C:/Users/User/OneDrive/Pictures/ม้วนฟิล์ม/WIN_20250817_12_08_11_Pro.mp4")
# cap = cv2.VideoCapture("C:/Users/Thinkpad/Pictures/Camera Roll/WIN_20250819_15_45_18_Pro.mp4")

def predicing(data):
    landmarks_np = np.array(data).reshape(1, -1)
    pred = model.predict(landmarks_np)
    index = np.argmax(pred)
    for id, confidence in enumerate(pred[0]):
        print(f"{class_names[id]} {round((confidence*100),2)}")
    print(f"final: {class_names[index]}")

path = "Main_video/tilted"
for classs in os.listdir(path):
    path2 = os.path.join(path, classs)
    allvideo = os.listdir(path2)
    for id in order:
        final_path = os.path.join(path2, allvideo[id])
        cap = cv2.VideoCapture(final_path)
        print(f"now predict class: {classs}")
        predicing(main_save())
        if input("Enter 0 to continue:") == "0":
            pass
        else:
            break