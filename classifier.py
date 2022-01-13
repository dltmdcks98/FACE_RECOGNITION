#이미지 처리를 위함
import cv2
import numpy as np
import face_recognition

#시각화를 위함
from matplotlib import pyplot as plt

#파일 선택 창을 위함
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import warnings

#파일 선택 창 생성
Tk().withdraw()
file = askopenfilename()
print(file)

#이미지 불러오기
image = cv2.imread(file)

plt.figure(figsize=(12,8))
plt.imshow(image)
plt.xticks([]), plt.yticks([]) #x,y측 숨김
plt.show

#흑백으로 전환
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(grayImage)
print("change succes")

#흑백 이미지 그리기 
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([]) #x,y측 숨김
plt.show

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)
print(faces.shape)
print("face: " + str(faces.shape[0]))

