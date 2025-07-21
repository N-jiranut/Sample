import mediapipe,cv2,time,numpy
from tensorflow.keras.models import load_model
pose_pic = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()

cap = cv2.VideoCapture(0)

model = load_model("ML-model/6-15-2025-model01.h5")
with open("ML-model/6-15-2025-label01.txt", "r") as f:
    class_names = f.read().splitlines()

while True:
    ret, image = cap.read()
    if not ret:
        print("cant find camera")
        break
    img = cv2.flip(image, 1)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    results_pose = pose_pic.process(img)
    results_hand = hands.process(img)
    row=[]
    LeftHand=[]
    RightHand = []
    pose = []
    if results_hand.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
            handedness = results_hand.multi_handedness[idx].classification[0].label
            for id, lm in enumerate(hand_landmarks.landmark):
                if handedness:                        
                    if handedness == "Left" and len(LeftHand) < 63:
                        LeftHand.extend([lm.x,lm.y,lm.z])
                    elif handedness == "Right" and len(RightHand) < 63:
                        RightHand.extend([lm.x,lm.y,lm.z])
    if len(LeftHand) == 0:
        LeftHand=[0 for n in range(63)]     
    if len(RightHand) == 0:
        RightHand=[0 for n in range(63)]

    row.extend(LeftHand)
    row.extend(RightHand)
    
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        for lm in landmarks:
            pose.extend([lm.x, lm.y, lm.z])
    if len(pose) == 0:
        pose=[0 for n in range(99)]  
    row.extend(pose)    
    
    if len(row) == 225:
        landmarks_np = numpy.array(row).reshape(1, -1)
        pred = model.predict(landmarks_np)
        index = numpy.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])
    
    cv2.imshow("Cheese",img)
    print(len(row))
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    time.sleep(.1)

cap.release()
cv2.destroyAllWindows()

# import csv
# with open("data/sss.csv") as file:
#     row = csv.reader(file)
#     row = list(row)
    
#     # print(row[0][-1])
#     readed_list=[]
#     readed_list.append(0)
#     for n in row[0]:
#         if n == 'ï»¿0':
#             n = n[3:]
        
#         try:
#             readed_list.append(int(n))
#         except:
#             readed_list.append(float(n))
#     # print(type(readed_list))

# landmarks_np = numpy.array(readed_list).reshape(1, -1)
# pred = model.predict(landmarks_np)
# index = numpy.argmax(pred)
# label = class_names[index]
# print("Prediction:", label, "Confidence:", pred[0][index])