import cv2
import numpy as np
import face_recognition


from tkinter import Tk
from tkinter.filedialog import askopenfilename

#파일 선택 창 생성

#Elon
Tk().withdraw()
Elon_file = askopenfilename()
print(Elon_file)

#TEST
Tk().withdraw()
Elon_test = askopenfilename()
print(Elon_test)

#Bill
Tk().withdraw()
Bill_file = askopenfilename()
print(Bill_file)


#이미지 출력
imgElon = face_recognition.load_image_file(Elon_file)
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2GRAY)

imgTEST = face_recognition.load_image_file(Elon_test)
imgTEST = cv2.cvtColor(imgTEST,cv2.COLOR_BGR2RGB)

imgBill = face_recognition.load_image_file(Bill_file)
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)


#=======이미지 분석
#Elon
faceLocElon = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocElon[3], faceLocElon[0]), (faceLocElon[1], faceLocElon[2]), (255, 0, 255), 2)

#Elon_TEST 
faceLocTEST = face_recognition.face_locations(imgTEST)[0]
encodeTEST = face_recognition.face_encodings(imgTEST)[0]
cv2.rectangle(imgTEST,(faceLocTEST[3], faceLocTEST[0]), (faceLocTEST[1], faceLocTEST[2]), (255, 0, 255), 2)

#Bill
faceLocBill = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLocBill[3], faceLocBill[0]), (faceLocBill[1], faceLocBill[2]), (255, 0, 255), 2)


#얼굴 비교 /Elon
results1 = face_recognition.compare_faces([encodeElon], encodeTEST)
results2 = face_recognition.compare_faces([encodeElon], encodeBill)
print('Elon + Elon_TEST : ', results1)
print('Elon + Bill : ', results2)

faceDis1 = face_recognition.face_distance([encodeElon], encodeTEST)
faceDis2 = face_recognition.face_distance([encodeElon], encodeBill)

print('Elon + Elon_TEST : %s (%f%%)' %(results1, 1-faceDis1))
print('Elon + Bill : %s (%f%%)' %(results2, 1-faceDis2))

cv2.imshow('Elon', imgElon)
cv2.imshow('Elon_TEST',imgTEST)
cv2.imshow('Bill', imgBill)
cv2.waitKey(0)