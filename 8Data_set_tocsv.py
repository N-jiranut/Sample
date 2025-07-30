import os, cv2, time
import mediapipe as mp
import pandas as pd
import numpy as np
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()

data = []
main = []
pose_take = [0,11,12,13,14,15,16]

def show_left(location):
    black_screenL = np.zeros((480, 480, 3), dtype=np.uint8)
    
    orgX, orgY = location[9]     
    offX = 240 - orgX
    offY = 240 - orgY 
                        
    for id in range(len(location)):
        lmx, lmy = location[id] 
        cv2.circle(black_screenL, (lmx+offX,lmy+offY), 1, (0,0,255), 7)  
    cv2.imshow("CropL", black_screenL) 
def show_right(location):
    black_screenR = np.zeros((480, 480, 3), dtype=np.uint8)
    
    orgX, orgY = location[9]     
    offX = 240 - orgX
    offY = 240 - orgY 
                        
    for id in range(len(location)):
        lmx, lmy = location[id] 
        cv2.circle(black_screenR ,(lmx+offX,lmy+offY), 1, (0,0,255), 7)  
    cv2.imshow("CropR", black_screenR) 

def black_canvas(image,rounded):
    LH=[]
    RH=[]
    row=[]
    img = cv2.imread(image)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(img)
    pose_results = pose.process(img)
    
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            point_show = []
            handedness = hand_result.multi_handedness[idx].classification[0].label
            for id, lm in enumerate(hand_landmarks.landmark): 
                x, y= lm.x, lm.y
                point_show.append([round(x*480),round(y*480)])           
                if handedness == "Left" and len(LH)<42:
                    LH.extend([x,y])
                if handedness == "Right" and len(RH)<42:
                    RH.extend([x,y])
            if handedness == "Left":
                show_left(point_show)
            if handedness == "Right":              
                show_right(point_show) 
            
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
                cv2.circle(img ,(round(x*600),round(y*480)), 1, (0,0,255), 7)
                row.extend([x, y])    
    else:
        row.extend([0 for _ in range(14)])
    
    row.append(rounded)
    
    main.append(row)    
    
    cv2.imshow("Real", img)        
    cv2.waitKey(1)

path = "One-Stage-TFS Thai One-Stage Fingerspelling Dataset"

onesf = os.listdir(path)

for item in onesf:
    if item == "Training set":
        item_path = os.path.join(path, item)
        secoundsf = os.listdir(item_path)
        for iitem in secoundsf:
            iitem_path = os.path.join(item_path, iitem, "Images (JPEG)")
            thirdsf = os.listdir(iitem_path)
            for picture in thirdsf:
                picture_path = os.path.join(iitem_path, picture)
                try:
                    black_canvas(picture_path, iitem)
                except:
                    pass

# df = pd.DataFrame(main)
# df.to_csv("data/OMG.csv", mode='w', index=False, header=False) 
cv2.destroyAllWindows
print('end')