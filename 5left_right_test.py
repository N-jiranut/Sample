import mediapipe,cv2,time
import numpy as np
from tensorflow.keras.models import load_model
pose_pic = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()

cap = cv2.VideoCapture(0)
    
name = "M8-1-2025-LRmove"

model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()

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

while True:
    row=[]
    LH=[]
    RH=[]
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
            # print(handedness)
                
            for id, lm in enumerate(hand_landmarks.landmark): 
                x, y, z = lm.x, lm.y, lm.z
                if len(landmark_location)<42:
                    # landmark_location.extend([x,y])     
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
        
        # print(row)   
        landmarks_np = np.array(row).reshape(1, -1)
        pred = model.predict(landmarks_np)
        index = np.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])   
    # time.sleep(.5)    
        
    cv2.imshow("Real", img)        
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    