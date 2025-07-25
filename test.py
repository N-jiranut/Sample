import cv2, mediapipe, time
hands = mediapipe.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

def test1(data):
    print(data)
def test2(data):
    print(data)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(img)
        
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            handedness = hand_result.multi_handedness[idx].classification[0].label
            # print(handedness)
            if handedness == "Left":
                print(0)
                test1(handedness)
            if handedness == "Right":
                print(1)
                test2(handedness)
            print("===")
    print("OOOOOOOOO")
    time.sleep(0.25)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()