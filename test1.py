import cv2 
cap = cv2.VideoCapture(0)
n=0
while True:
    n+=1
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow("Preview", img)
    cv2.imwrite(f"clint_picture/img{n}.png", img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()