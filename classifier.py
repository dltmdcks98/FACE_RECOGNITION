#�̹��� ó���� ����
import cv2
import numpy as np
#�ð�ȭ�� ����
from matplotlib import pyplot as plt

#�̹��� �ҷ�����
image = cv2.imread("C:/Users/dltmd\Downloads/IU_TEST/[1752705] DSC_4979.jpg")

#������� ��ȯ
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#��� �̹��� �׸��� 
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([]) #x,y�� ����
plt.show
