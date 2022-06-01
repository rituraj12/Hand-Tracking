import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # number 3 is for width
cap.set(4, hCam)  # number 2 is for height

folderPath = "D:\PROGRAMMING\Projects\HandTracking\pic" 
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIDs = [4, 8, 12, 16, 20]
totalFingers = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)     # makes a list of coordinates of the points in our hand
    #print(lmlist)

    if len(lmlist) != 0:
        totalFingers = []
        fingers = []

        # Thumb
        if lmlist[tipIDs[0]][1] > lmlist[tipIDs[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmlist[tipIDs[id]][2] < lmlist[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

    h, w, c = overlayList[totalFingers-1].shape
    img[0:h, 0:w] = overlayList[totalFingers-1]

    cv2.rectangle(img, (20, 225), (170, 425), (0,255,0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 20)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break