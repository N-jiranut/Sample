import cv2, os, math
from flask import Flask, render_template, Response, request
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
name = "M8-17-2025-moving_hands_full"
name = "M8-21-2025-moving_hands_full"
model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()
pose_take = [0,11,12,13,14,15,16]
path = "client_picture"
output = ""
processing = False
value = False
npic = 0

test = 0

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # 0 = default webcam

def generate_frames():
    global npic, output
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot find camera")
            break
        else:
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


            if value and npic < 200:
                forsave = frame
                cv2.imwrite(f"client_picture/img{npic}.jpeg", forsave)
                cv2.putText(frame, f"get:{npic}/200 pic", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                # output = "collecting.."
                npic+=1


            hand_result = hands.process(frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            if hand_result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    arx, ary = [], []
                    handedness = hand_result.multi_handedness[idx].classification[0].label
                    for lm in hand_landmarks.landmark: 
                        x, y= lm.x, lm.y    
                        arx.append(x)
                        ary.append(y)

                    mx = round(max(arx)*720)
                    lx = round(min(arx)*720)
                    my = round(max(ary)*480)
                    ly = round(min(ary)*480)

                    cv2.rectangle(frame, (lx,ly), (mx,my),(50,150,0), 2)
                    cv2.putText(frame, str(handedness), (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use yield with multipart/x-mixed-replace for live stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def forpredict():
    global path, processing, output
    processing = True
    pose_take = [0,11,12,13,14,15,16]
    picture = os.listdir(path)
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
                    BO = ([0 for _ in range(14)])
                row.extend(BO)
                con += 1

            # cv2.imshow("screen", cimg)
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
    processing = False
    output = label
    # return label

@app.route('/', methods=["GET", "POST"])
def index():
    global value, npic, output, processing
    if request.method == "POST" and not processing:
        if value:
            value=False
            forpredict()
        elif not value:
            npic = 0
            value=True
            fordel = os.listdir(path)
            for pic in fordel:
                file_path = os.path.join(path, pic)
                os.remove(file_path)
    return render_template('index.html', result = output)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/laptop')
def get_notifications():
    return render_template('demo web interface.html')
    

if __name__ == "__main__":
    app.run(debug=True)