import cv2
import numpy as np
import mediapipe as mp

type = ["one", "two", "three", "four", "five"]

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
focus = 9

def black_canvas(frame,classs,mode,round):
    for i in range(frame):
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
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    if handedness == "Right":
                        for id, lm in enumerate(hand_landmarks.landmark): 
                            x, y, z = lm.x, lm.y, lm.z
                            landmark_location.append([x,y])                        
                            mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

        if len(landmark_location)>0:     
            cv2.imshow("Crop", black_screen) 
            if int(mode) == 0: 
                cv2.imwrite(f"data/test/saved{round}.jpg",black_screen) 
            else:
                cv2.imwrite(f"data/{classs}/saved{i}.jpg",black_screen) 
        
        cv2.imshow("Real", img)        
        cv2.waitKey(1)


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

for i in range(5):
    wait()
    black_canvas(1, None, 0, i)
    
cv2.destroyAllWindows
cap.release
print('end')