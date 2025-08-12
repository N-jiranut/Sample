import cv2, time
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pca = PCA(n_components=100)
pose_take = [0,11,12,13,14,15,16]

name = "M8-11-2025-moving_hands"

model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()

print("Here")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

def main_save():
    row=[]
    condition=0
    con = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if not ret:
            break
        if con == 15:
            break
        if condition < 20:
            pass
        elif condition%5 == 0 and con < 15:
            print(con)
            hand_result = hands.process(frame)
            pose_results = pose.process(frame)
            LH=[]
            BO=[]
            RH=[]
            if hand_result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                    
                    for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                        handedness = hand_result.multi_handedness[idx].classification[0].label
                        for id, lm in enumerate(hand_landmarks.landmark): 
                            x, y= lm.x, lm.y         
                            if handedness == "Left" and len(LH)<42:
                                LH.extend([x,y])
                            if handedness == "Right" and len(RH)<42:
                                RH.extend([x,y])
                        
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    for id,lm in enumerate(landmarks):
                        x, y = lm.x, lm.y
                        if id in pose_take:
                            cv2.circle(frame ,(round(x*600),round(y*480)), 1, (0,0,255), 7)
                            BO.extend([x, y])

            if len(LH) == 0:
                LH = [0 for _ in range(42)]
            row.extend(LH)
            if len(RH) == 0:
                RH = [0 for _ in range(42)]
            row.extend(RH)
            if len(BO) == 0:
                BO = [0 for _ in range(14)]
            row.extend(BO)
            con += 1

        cv2.putText(frame, f"Get {con+1}/15", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,200,0), 3)         
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord('q'):
           break
        condition+=1
    
    if con < 15:
        row.extend([0 for _ in range(98*(15-con))])
    return row

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow("Video",frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        data = main_save()
        
        data = pca.fit_transform(data)
        print(len(data))
        # data.append(main_save())

        landmarks_np = np.array(data).reshape(1, -1)
        pred = model.predict(landmarks_np)
        index = np.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])

    if key == ord("q"):
        break

# row = [0 for _ in range(1470)]
# landmarks_np = np.array(row).reshape(1, -1)
# pred = model.predict(landmarks_np)
# index = np.argmax(pred)
# label = class_names[index]
# print("Prediction:", label, "Confidence:", pred[0][index])