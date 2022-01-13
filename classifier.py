import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/IU_TEST.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgBill = face_recognition.load_image_file('ImageBasic/BAEK.jpg')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Bill Gates', imgBill)
cv2.waitKey(0)