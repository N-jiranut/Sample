import cv2,time
import mediapipe as mp
hands = mp.solutions.hands.Hands()

# video = "C:/Users/Thinkpad/Documents/Github/Sample/main_video/ทำไม/WIN_20250806_16_34_12_Pro.mp4"
video = "main_video/ทำไม/WIN_20250806_16_37_52_Pro.mp4"

collect = 0
find = 0

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    print(collect)
    ret, frame = cap.read()
    if not ret:
        break
    if collect%1 == 0:
        try:
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            hand_result = hands.process(frame)

            if hand_result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
            cv2.imshow("Video", frame)
        except:
            pass
        find += 1 
    if cv2.waitKey(1) == ord('q'):
        break
    collect+=1
    time.sleep(.1)

print(f"Total frame :{find}")
cap.release()
cv2.destroyAllWindows()
