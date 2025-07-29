import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

type = ["one", "two", "three", "four", "five"]

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
main = []

def black_canvas(frame,classs,mode,rounded):
    for i in range(frame):
        row=[]
        landmark_location = []
        point_show = []
        black_screen = np.zeros((480, 480, 3), dtype=np.uint8)
        ret, img = cap.read()
        if not ret:
            print("Cam not found.")
            break
        img = cv2.flip(img, 1)
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        hand_result = hands.process(img)
        
        if hand_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    if handedness == "Right":
                        for id, lm in enumerate(hand_landmarks.landmark): 
                            x, y, z = lm.x, lm.y, lm.z
                            if len(landmark_location)<21:
                                landmark_location.append([x,y])     
                                point_show.append([round(x*480),round(y*480)])           

            orgX, orgY = point_show[9]     
            offX = 240 - orgX
            offY = 240 - orgY 
            
            for id in range(len(point_show)):
                lmx, lmy = point_show[id] 
                cv2.circle(black_screen, (lmx+offX,lmy+offY), 1, (0,0,255), 7)     
            
        if len(landmark_location)>0:   
            for cor in landmark_location:
                row.extend(cor)

            row.append(rounded)
            main.append(row)
            cv2.imshow("Crop", black_screen) 

        else:
            main.append(0 for _ in range(21))
            main.append(rounded)
        cv2.imshow("Real", img)        
        cv2.waitKey(1)
        # print(row)

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

# for types in type:
#     wait()
#     black_canvas(10, types, 1, 1)

for i in type:
    wait()
    black_canvas(5, None, 0, i)

# print(main)
print(len(main))

df = pd.DataFrame(main)
df.to_csv("data/testing.csv", mode='w', index=False, header=True) 
cv2.destroyAllWindows
cap.release
print('end')