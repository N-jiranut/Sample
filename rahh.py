import os
import cv2 # OpenCV
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = "One-Stage-TFS Thai One-Stage Fingerspelling Dataset/Training" # เปลี่ยนเป็นพาธที่คุณแตกไฟล์
images = []
labels = []
class_names = sorted(os.listdir(data_dir)) # ก, ข, ค, ...

for i, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name, "Images (JPEG)")
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64)) # ปรับขนาดภาพให้เท่ากัน
        images.append(img)
        labels.append(i) # เก็บเป็นตัวเลข 0, 1, 2...

images = np.array(images) / 255.0 # แปลงเป็น NumPy array และปรับ scale ค่าสี (0-1)
labels = to_categorical(np.array(labels)) # แปลง label เป็น One-Hot Encoding