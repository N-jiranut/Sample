import mediapipe,cv2,time,numpy
from tensorflow.keras.models import load_model
pose_pic = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()

cap = cv2.VideoCapture(0)

model = load_model("ML-model/M7-23-2025-test.h5")
with open("ML-model/M7-23-2025-test.txt", "r") as f:
    class_names = f.read().splitlines()

while True:
    row=[]
    landmark_location = []
    point_show = []
    black_screen = numpy.zeros((480, 480, 3), dtype=numpy.uint8)
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
                    if len(landmark_location)<21:
                        landmark_location.append([x,y])     
                        point_show.append([round(x*480),round(y*480)])           
                        # mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
        try:
            orgX, orgY = point_show[9]     
            offX = 240 - orgX
            offY = 240 - orgY 
                
            for id in range(len(point_show)):
                lmx, lmy = point_show[id] 
                cv2.circle(black_screen, (lmx+offX,lmy+offY), 1, (0,0,255), 7)     
        except:
            pass
            
    if len(landmark_location)>0:   
        for cor in landmark_location:
            row.extend(cor)
        cv2.imshow("Crop", black_screen) 

    else:
        row.append(0 for _ in range(21))
      
        
    if len(row) > 5:
        print("Working")
        landmarks_np = numpy.array(row).reshape(1, -1)
        # landmarks_np = numpy.array(row)
        # print(landmarks_np)
        pred = model.predict(landmarks_np)
        index = numpy.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])   

    cv2.imshow("Real", img)        
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()