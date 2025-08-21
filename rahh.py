import os, cv2

path = "Main_video/tilted"
for classs in os.listdir(path):
    path2 = os.path.join(path, classs)
    for video in os.listdir(path2):
        final_path = os.path.join(path2, video)
        # cap = cv2.VideoCapture(final_path)
        print(final_path)
        if input("Enter 0 to continue:") == "0":
            pass
        else:
            break