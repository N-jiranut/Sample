import cv2, os, math, time
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
path = "client_picture"
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]

name = "M8-17-2025-moving_hands"
model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()

picture = os.listdir(path)

def forpredict():
    LH = ["L"]
    RH = ["R"]
    BO = ["B"]
    row=[]
    condition=0
    con = 0
    allcimg = len(picture)    
    percimg = math.floor(allcimg/15)
    for img in picture:
        current_img_path = os.path.join(path, img)
        cimg = cv2.imread(current_img_path)
        if condition%percimg == 0 and con < 15:
            cimg = cv2.resize(cimg, (720, 480))
            hand_result = hands.process(cimg)
            pose_results = pose.process(cimg)
            if hand_result.multi_hand_landmarks:
                RH=[]
                LH=[]
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(cimg,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                    handedness = hand_result.multi_handedness[idx].classification[0].label      
                    if handedness == "Left":
                        LH.extend("L")
                    elif handedness == "Right":
                        RH.extend("R")
                if len(LH) <= 0:
                    LH = ["L"]
                row.extend(LH)
                if len(RH) <= 0:
                    RH = ["R"]
                row.extend(RH)

                if pose_results.pose_landmarks:
                    BO=[]
                    BO.extend("B")    
                else:
                    BO=["BI"]
                row.extend(BO)
                con += 1
                
            time.sleep(.05)
            cv2.imshow("Video", cimg)
            
        if cv2.waitKey(1) == ord('q'):
           break
        condition+=1

    if con < 15 and len(row) < 1470:
        foradd=[]
        for object in [LH,RH,BO]:
            foradd.extend(object)
        row.extend(foradd*(15-con))

    # landmarks_np = np.array(row).reshape(1, -1)
    # pred = model.predict(landmarks_np)
    # index = np.argmax(pred)
    # label = class_names[index]

    # print(pred[0][index])
    # print(label)
    print(row)
    testss=0
    for obj in row:
        if obj == "L":
            testss += 1
    print(testss)

forpredict()
cv2.destroyAllWindows()