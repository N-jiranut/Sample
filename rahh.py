# import cv2

# path = "One-Stage-TFS Thai One-Stage Fingerspelling Dataset/Training set/BOR_BAI_MAI/Images (JPEG)/BOR_BAI_MAI_2.jpg"

# pic = cv2.imread(path)

# cv2.imshow("test", pic)

# cv2.waitKey(0)
import os 

mp = "test_folder/pic"

test = os.listdir(mp)
print(test)