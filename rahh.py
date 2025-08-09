import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

name = ""

path = ""

save = False
label = 0
while True:
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    key = cv2.waitKey(1)
    if key == ord("s"):
        if save == True:
            save = False
        else:
            save=True
    if key == ord("q"):
        break