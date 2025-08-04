import cv2
import numpy as np

screen = np.zeros((480, 480, 3), dtype=np.uint8)

cv2.rectangle(screen, (0,480), (10,470),(50,150,0), 2)

cv2.imshow("test",screen)

cv2.waitKey(0)