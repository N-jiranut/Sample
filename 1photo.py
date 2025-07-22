import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

type = ["one", "two", "three", "four", "five"]

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
main = []

def black_canvas(frame,classs,mode,round):
    for i in range(frame):
        row=[]
        landmark_location = []
        test = []
        black_screen = np.zeros((480, 480, 3), dtype=np.uint8)
        ret, img = cap.read()
        if not ret:
            print("Cam not found.")
            break
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        hand_result = hands.process(img)
        
        if hand_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    if handedness == "Right":
                        for id, lm in enumerate(hand_landmarks.landmark): 
                            x, y, z = lm.x, lm.y, lm.z
                            if len(test)<21:
                                test.append([x,y])                        
                            mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

        if len(landmark_location)>0:   
            for cor in landmark_location:
                row.extend(cor)
            row.append(round)
            main.append(row)
            cv2.imshow("Crop", black_screen) 
            # if int(mode) == 0: 
            #     cv2.imwrite(f"data/test/saved{round}.jpg",black_screen) 
            # else:
            #     cv2.imwrite(f"data/{classs}/saved{i}.jpg",black_screen)

        else:
            main.append(0 for _ in range(21))
            main.append(round)
        cv2.imshow("Real", img)        
        cv2.waitKey(1)
    # print(len(main))
    # df = pd.DataFrame(main)
    # df.to_csv("data/main.csv", mode='a', index=False, header=False) 


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

for i in range(3):
    wait()
    black_canvas(20, None, 0, i)

# print(main)
# print(len(main))
df = pd.DataFrame(main)
df.to_csv("data/main.csv", mode='w', index=False, header=False) 
cv2.destroyAllWindows
cap.release
print('end')