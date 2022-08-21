import cv2
import numpy as np
import os 
import sys
import time

# currently this file is used to test recognizer

from user_retrieve import *
sys.path.append(os.path.join(sys.path[0], "Face_Detection"))
from Face_Detection.capture import *

def beginFacialRecognition(currUser):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    recognizer.read('Facial_Recognition\\trainer.yml')
    face_classifier = cv2.CascadeClassifier(os.path.join(sys.path[0], "Face_Detection\\faceCascade.xml"))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = FaceDetector(currUser)
    userDB = UserData()
    userDB.loadUsernames()
    users = userDB.getUsernamesList()
    cam.capture()
    confidence = 100
    while int(confidence) >= 40:
        img, grayImg, faces = cam.detectFaces()
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(grayImg[y:y+h,x:x+w])
            # Check if confidence is less them 100, "0" is perfect match 
            if (confidence < 100):
                id = users[id]
                confidenceStr = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidenceStr = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidenceStr), (x+5,y+h-5), font, 1, (255,255,0), 1) 
        cv2.imshow('camera', img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.capture()
    img, _, faces = cam.detectFaces()
    cv2.putText(img, "Welcome back " + str(currUser), (50,50), font, 2, (255,0,255), 2)
    cv2.imshow('camera', img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        sys.exit()
    time.sleep(1)
    # cleanup
    cv2.destroyAllWindows()




