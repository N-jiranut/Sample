import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
pose_take = [0,11,12,13,14,15,16]

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # 0 = default webcam

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

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot find camera")
            break
        else:
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            hand_result = hands.process(frame)
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
