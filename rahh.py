import cv2
import mediapipe as mp
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]

cap = cv2.VideoCapture(0)

def getminmax(array,side):
    arx, ary = [], []
    
    for twin in array:
        arx.append(twin[0])
        ary.append(twin[1])
    
    mx = round(max(arx)*720)
    lx = round(min(arx)*720)
    my = round(max(ary)*480)
    ly = round(min(ary)*480)
    
    cv2.rectangle(frame, (lx,ly), (mx,my),(50,150,0), 2)
    cv2.putText(frame, str(side), (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(frame)
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            lmlist=[]
            handedness = hand_result.multi_handedness[idx].classification[0].label
            for lm in hand_landmarks.landmark: 
                x, y= lm.x, lm.y    
                lmlist.append([x,y])
            getminmax(lmlist, handedness)
                     
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()