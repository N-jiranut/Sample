import cv2,time
import mediapipe as mp 
hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture("Main_video/tilted/me/test.mp4")
while cap.isOpened:
    ret, img = cap.read()
    img = cv2.resize(img, (720, 480))
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(img)
    if hand_result.multi_hand_landmarks:
        print("Ser")
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(img,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark): 
                x, y= lm.x, lm.y     
    
    cv2.imshow("test",img)
    
    cv2.waitKey(1)
    time.sleep(.25)