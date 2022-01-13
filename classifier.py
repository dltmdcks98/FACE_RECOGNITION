#�̹��� ó���� ����
from dataclasses import replace
import cv2
import numpy as np
#�ð�ȭ�� ����
from matplotlib import pyplot as plt

#���� ���� â�� ����
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#���� ���� â ����
Tk().withdraw()
file = askopenfilename()
print(file)

#�̹��� �ҷ�����
image = cv2.imread(file)

#������� ��ȯ
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#��� �̹��� �׸��� 
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([]) #x,y�� ����
plt.show
