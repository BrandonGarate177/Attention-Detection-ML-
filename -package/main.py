import cv2
import os
import numpy as np
import face_recognition 

img = cv2.imread("/Users/brandon/Programming Projects./Facial-Recognition/-package/reference.png")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_encodings = face_recognition.face_encodings(rgb_img)[0]

cv2.imshow("Img", img)
cv2.waitKey(0)


