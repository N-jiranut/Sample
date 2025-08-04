import cv2, time
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
hands = mp.solutions.hands.Hands()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
main = []

name = "M8-4-2025-LRmove"

model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()

def wait():
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        hand_result = hands.process(img)
        
        if hand_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(img,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                
        cv2.imshow("test", img)
        if cv2.waitKey(1) == ord("q"):
            break

def black_canvas(FPS):
    while True:
        row=[]
        wait()
        for frame in range(FPS):
            print(f"{frame+1}/{FPS}")
            LH=[]
            RH=[]
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            hand_result = hands.process(img)

            if hand_result.multi_hand_landmarks:
                DR, DL = False, False
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    for id, lm in enumerate(hand_landmarks.landmark): 
                        x, y= lm.x, lm.y          
                        if handedness == "Left" and len(LH)<42:
                            DL=True
                            NX = round(x*600)
                            NY = round(y*480)
                            if id == 0:
                                LXM, LXL, LYM, LYL = NX,NX,NY,NY
                            if NX>LXM:
                                LXM=NX+40
                            elif NX<LXL:
                                LXL=NX
                            if NY>LYM:
                                LYM=NY
                            elif NY<LYL:
                                LYL=NY
                        if handedness == "Right" and len(RH)<42:
                            DR=True
                            NX = round(x*600)
                            NY = round(y*480)
                            if id == 0:
                                RXM, RXL, RYM, RYL = NX,NX,NY,NY
                            if NX>RXM:
                                RXM=NX+40
                            elif NX<RXL:
                                RXL=NX
                            if NY>RYM:
                                RYM=NY
                            elif NY<RYL:
                                RYL=NY
                            RH.extend([x,y])
                if DR:
                    cv2.rectangle(img, (RXL,RYL), (RXM,RYM),(50,150,0), 2)
                if DL:
                    cv2.rectangle(img, (LXL,LYL), (LXM,LYM),(50,150,0), 2)

            if len(RH) <= 0:
                RH = [0 for _ in range(42)]
            row.extend(RH)   

            cv2.imshow("Real", img)        
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        landmarks_np = np.array(row).reshape(1, -1)
        pred = model.predict(landmarks_np)
        index = np.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])   

black_canvas(10)

cv2.destroyAllWindows
print('end')