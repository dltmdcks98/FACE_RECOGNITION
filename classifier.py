import cv2
import numpy as np
import face_recognition

from tkinter import Tk
from tkinter.filedialog import askopenfilename

#파일 선택 창 생성

#IU
Tk().withdraw()
IU_file = askopenfilename()
print(IU_file)

#TEST
Tk().withdraw()
IU_test = askopenfilename()
print(IU_test)

#BAEK
Tk().withdraw()
BAEK_file = askopenfilename()
print(BAEK_file)


#이미지 출력
imgIU = face_recognition.load_image_file(IU_file)
imgIU = cv2.cvtColor(imgIU, cv2.COLOR_BGR2RGB)

imgTEST = face_recognition.load_image_file(IU_test)
imgTEST = cv2.cvtColor(imgTEST,cv2.COLOR_BGR2RGB)

imgBAEK = face_recognition.load_image_file(BAEK_file)
imgBAEK = cv2.cvtColor(imgBAEK, cv2.COLOR_BGR2RGB)


#=======이미지 분석
#IU
faceLocIU = face_recognition.face_locations(imgIU)[0]
encodeIU = face_recognition.face_encodings(imgIU)[0]
cv2.rectangle(imgIU, (faceLocIU[3], faceLocIU[0]), (faceLocIU[1], faceLocIU[2]), (255, 0, 255), 2)

#IU_TEST 
faceLocTEST = face_recognition.face_locations(imgTEST)[0]
endcodeTEST = face_recognition.face_encodings(imgTEST)[0]
cv2.rectangle(imgTEST,(faceLocTEST[3], faceLocTEST[0]), (faceLocTEST[1], faceLocTEST[2]), (255, 0, 255), 2)

#BAEK
faceLocBAEK = face_recognition.face_locations(imgBAEK)[0]
encodeBEAK = face_recognition.face_encodings(imgBAEK)[0]
cv2.rectangle(imgBAEK,(faceLocBAEK[3], faceLocBAEK[0]), (faceLocBAEK[1], faceLocBAEK[2]), (255, 0, 255), 2)


#얼굴 비교 /IU
results1 = face_recognition.compare_faces([encodeIU], endcodeTEST)
results2 = face_recognition.compare_faces([encodeIU], encodeBEAK)
print('IU + IU_TEST : ', results1)
print('IU + BAEK : ', results2)

cv2.imshow('IU', imgIU)
cv2.imshow('IU_TEST',imgTEST)
cv2.imshow('BAEK', imgBAEK)
cv2.waitKey(0)