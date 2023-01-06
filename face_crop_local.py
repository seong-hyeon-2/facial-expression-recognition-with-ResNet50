import cv2
import numpy as np
import os

# 감정 지정
emotion = "emotion"

# 이미지 경로 가져오기
path_dir = f"User_Path/{emotion}/"
file_list = os.listdir(path_dir)

# file_list[0]

# len(file_list)

def Cutting_face_save(image, num):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1.3 값을 더 높이면 프레임이 더 넓어짐
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (96, 96))

        # 이미지 저장하기
        cv2.imwrite(f"User_Path/crop_{emotion}/{emotion}_{num}.jpg", resize)

for i in range(len(file_list)):
    image = cv2.imread(path_dir+file_list[i])
    Cutting_face_save(image, i)