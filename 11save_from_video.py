import cv2,time,os
import mediapipe as mp
import pandas as pd
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]
main_data = []

def main_save(classs):
    row=[]
    condition=0
    con = 0
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        if condition < 20:
            pass
        elif condition%5 == 0 and con < 15:
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            hand_result = hands.process(frame)
            pose_results = pose.process(frame)
            if hand_result.multi_hand_landmarks:
                LH=[]
                RH=[]
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
                        
                if len(LH) <= 0:
                    LH = [0 for _ in range(42)]
                row.extend(LH)
                if len(RH) <= 0:
                    RH = [0 for _ in range(42)]
                row.extend(RH)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    for id,lm in enumerate(landmarks):
                        x, y = lm.x, lm.y
                        if id in pose_take:
                            cv2.circle(frame ,(round(x*600),round(y*480)), 1, (0,0,255), 7)
                            row.extend([x, y])    
                else:
                    row.extend([0 for _ in range(14)])
                con += 1
                 
                cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord('q'):
           break
        condition+=1
    
    if con < 15:
        row.extend([0 for _ in range(98*(15-con))])
    row.append(classs)
    return row

path = "Main_video"

onefloor = os.listdir(path)
for posetype in onefloor:
    if posetype == "tilted":
        path2 = os.path.join(path, posetype)
        twofloor = os.listdir(path2)
        for poseclass in twofloor:
            path3 = os.path.join(path2, poseclass)
            threefloor = os.listdir(path3)
            for video in threefloor:
                path4 = os.path.join(path3, video)
                cap = cv2.VideoCapture(path4)
                if not cap.isOpened():
                    print("Error: Could not open video file.")
                    exit()
                
                main_data.append(main_save(poseclass))
                
df = pd.DataFrame(main_data)
df.to_csv("data/Moving_hands.csv", mode='w', index=False, header=False)                 
cap.release()
cv2.destroyAllWindows()