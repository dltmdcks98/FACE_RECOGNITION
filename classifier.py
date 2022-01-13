import cv2
import numpy as np
import face_recognition

from tkinter import Tk
from tkinter.filedialog import askopenfilename

#파일 선택 창 생성
Tk().withdraw()
IU_file = askopenfilename()
print(IU_file)

#이미지 출력
imgIU = face_recognition.load_image_file(IU_file)
imgIU = cv2.cvtColor(imgIU, cv2.COLOR_BGR2RGB)

Tk().withdraw()
BAEK_file = askopenfilename()
print(BAEK_file)

imgBAEK = face_recognition.load_image_file(BAEK_file)
imgBAEK = cv2.cvtColor(imgBAEK, cv2.COLOR_BGR2RGB)


#얼굴 인식
faceLoc = face_recognition.face_locations(imgIU)[0]
encodeElon = face_recognition.face_encodings(imgIU)[0]
cv2.rectangle(imgIU, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

cv2.imshow('IU', imgIU)
cv2.imshow('BAEK', imgBAEK)
cv2.waitKey(0)