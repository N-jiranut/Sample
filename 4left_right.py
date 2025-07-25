import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

type = ["one", "two", "three", "four", "five"]

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
main = []

def show_left(location):
    # print(location)
    black_screenL = np.zeros((480, 480, 3), dtype=np.uint8)
    
    orgX, orgY = location[9]     
    offX = 240 - orgX
    offY = 240 - orgY 
                        
    for id in range(len(location)):
        lmx, lmy = location[id] 
        cv2.circle(black_screenL, (lmx+offX,lmy+offY), 1, (0,0,255), 7)  
    cv2.imshow("CropL", black_screenL) 
def show_right(location):
    # print(location)
    black_screenR = np.zeros((480, 480, 3), dtype=np.uint8)
    
    orgX, orgY = location[9]     
    offX = 240 - orgX
    offY = 240 - orgY 
                        
    for id in range(len(location)):
        lmx, lmy = location[id] 
        cv2.circle(black_screenR ,(lmx+offX,lmy+offY), 1, (0,0,255), 7)  
    cv2.imshow("CropR", black_screenR) 

def black_canvas(frame,classs,mode,rounded):
    # for i in range(frame):
    while True:
        row=[]
        landmark_location = []
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
                point_show = []
                handedness = hand_result.multi_handedness[idx].classification[0].label
                    
                for id, lm in enumerate(hand_landmarks.landmark): 
                    x, y, z = lm.x, lm.y, lm.z
                    if len(landmark_location)<42:
                        landmark_location.append([x,y])     
                        point_show.append([round(x*480),round(y*480)])           
                        # mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

                if handedness == "Left":
                    # print("Left detect")
                    show_left(point_show)
                if handedness == "Right":
                    # print("Right detect")                        
                    show_right(point_show) 
                print(idx)
        print("OOOOOOOOOOOOO")    
        # if len(landmark_location)>0:   
        #     for cor in landmark_location:
        #         row.extend(cor)
        #         # print(row)
        #     row.append(rounded)
        #     main.append(row)

        # else:
        #     main.append(0 for _ in range(21))
        #     main.append(rounded)
        cv2.imshow("Real", img)        
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

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

for i in range(1):
    # wait()
    black_canvas(500, None, 0, i)

# print(main)
print(len(main))

# df = pd.DataFrame(main)
# df.to_csv("data/Left_Right.csv", mode='w', index=False, header=True) 
cv2.destroyAllWindows
cap.release
print('end')