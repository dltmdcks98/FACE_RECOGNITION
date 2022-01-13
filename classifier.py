#이미지 처리를 위함
import cv2
import numpy as np
#시각화를 위함
from matplotlib import pyplot as plt

#이미지 불러오기
image = cv2.imread("C:/Users/dltmd\Downloads/IU_TEST/[1752705] DSC_4979.jpg")

#흑백으로 전환
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#흑백 이미지 그리기 
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([]) #x,y측 숨김
plt.show
