import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

space = "Data/C"
cnt=0

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 30
imgSize = 500

while True:
    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgWite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCrpShp = imgCrop.shape

        ascept_ratio = h/w

        if ascept_ratio>1:
            x = imgSize/h
            wCal = math.ceil(x*w)
            imgreisxe = cv2.resize(imgCrop,(wCal,imgSize))
            imgreisxeSph = imgCrop.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWite[:,wGap:wCal+wGap]=imgreisxe

        else:
            x = imgSize/w
            hCal = math.ceil(x*h)
            imgreisxe = cv2.resize(imgCrop,(imgSize,hCal))
            imgreisxeSph = imgCrop.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWite[hGap:hCal+hGap,:]=imgreisxe



        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageBG",imgWite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        cnt+=1
        cv2.imwrite(f'{space}/Image_{time.time()}.jpg',imgWite)
        print('Image saved:',cnt)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()