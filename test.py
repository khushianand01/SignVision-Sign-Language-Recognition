# import cv2
# import numpy as np
# import math
# import time
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# space = "Data/C"
# cnt=0

# labels=['A','B','C']
# claddifier = Classifier('model/keras_model.h5','model/labels.txt')
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# offset = 30
# imgSize = 500

# while True:
#     success,img = cap.read()
#     hands,img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x,y,w,h = hand['bbox']
#         imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
#         imgWite = np.ones((imgSize,imgSize,3),np.uint8)*255

#         imgCrpShp = imgCrop.shape

#         ascept_ratio = h/w

#         if ascept_ratio>1:
#             x = imgSize/h
#             wCal = math.ceil(x*w)
#             imgreisxe = cv2.resize(imgCrop,(wCal,imgSize))
#             imgreisxeSph = imgCrop.shape
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWite[:,wGap:wCal+wGap]=imgreisxe
#             prediction, index = claddifier.getPrediction(img)
#             print(prediction,index)

#         else:
#             x = imgSize/w
#             hCal = math.ceil(x*h)
#             imgreisxe = cv2.resize(imgCrop,(imgSize,hCal))
#             imgreisxeSph = imgCrop.shape
#             hGap = math.ceil((imgSize-hCal)/2)
#             imgWite[hGap:hCal+hGap,:]=imgreisxe
#             prediction, index = claddifier.getPrediction(img)
#             print(prediction,index)




#     cv2.imshow("Image",img)
#     key = cv2.waitKey(1)

#     if key == ord('s'):
#         cnt+=1
#         cv2.imwrite(f'{space}/Image_{time.time()}.jpg',imgWite)
#         print('Image saved:',cnt)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

space = "Data/C"
cnt = 0

model = load_model("model/keras_model.h5")
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 30
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = imgInput.astype(np.float32) / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        prediction = model.predict(imgInput)
        index = np.argmax(prediction)
        label = labels[index]
        confidence = round(float(prediction[0][index]) * 100, 2)

        text = f"{label} ({confidence}%)"
        text_x = x
        text_y = y - 10 if y > 40 else y + h + 30  # Keep it inside the frame
        cv2.putText(imgOutput, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ASL Detection", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
