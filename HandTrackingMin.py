import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils     # used to draw the track of hand

pTime = 0
cTIme = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results)
    #print(results.multi_hand_landmarks)  # to check if something is detected or not

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:    # to determine how many hands
            for id, lm in enumerate(handLms.landmark):  # get the information within this hand (get the id number and the landmark information(x,y coordinate))
                #print(id, lm)   # each id has a corresponding landmark
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)   # position of the centre, multiplying because we need the values in pixels values instead of decimal places which is the ratio of the image
                print(id, cx, cy)
                if id == 0:     # detecting the landmark
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)     # draw the points in a hand, after adding hand_connections it shows the connection lines
            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break
    