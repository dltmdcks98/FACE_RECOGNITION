#�̹��� ó���� ����
import cv2
import numpy as np
import face_recognition

#�ð�ȭ�� ����
from matplotlib import pyplot as plt

#���� ���� â�� ����
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import warnings

#���� ���� â ����
Tk().withdraw()
file = askopenfilename()
print(file)

#�̹��� �ҷ�����
image = cv2.imread(file)

plt.figure(figsize=(12,8))
plt.imshow(image)
plt.xticks([]), plt.yticks([]) #x,y�� ����
plt.show

#������� ��ȯ
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(grayImage)
print("change succes")

#��� �̹��� �׸��� 
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([]) #x,y�� ����
plt.show

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)
print(faces.shape)
print("face: " + str(faces.shape[0]))

