import cv2, os, math
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
path = "clint_picture"
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]

name = "M8-17-2025-moving_hands"
model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()

picture = os.listdir(path)

def forpredict():
    LH = [0 for _ in range(42)]
    RH = [0 for _ in range(42)]
    BO = [0 for _ in range(14)]
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
                    for lm in hand_landmarks.landmark: 
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
                    BO=[]
                    landmarks = pose_results.pose_landmarks.landmark
                    for id,lm in enumerate(landmarks):
                        x, y = lm.x, lm.y
                        if id in pose_take:
                            cv2.circle(cimg ,(round(x*720),round(y*480)), 1, (0,0,255), 7)
                            BO.extend([x, y])    
                else:
                    BO.extend([0 for _ in range(14)])
                row.extend(BO)
                con += 1

            cv2.imshow("Video", cimg)
        if cv2.waitKey(1) == ord('q'):
           break
        condition+=1

    if con < 15:
        foradd=[]
        for object in [LH,RH,BO]:
            foradd.extend(object)
        row.extend(foradd*(15-con))

    landmarks_np = np.array(row).reshape(1, -1)
    pred = model.predict(landmarks_np)
    index = np.argmax(pred)
    label = class_names[index]

    print(pred[0][index])
    print(label)

cv2.destroyAllWindows()

forpredict()