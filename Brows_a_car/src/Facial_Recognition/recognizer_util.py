import cv2
import os
import sys

sys.path.append(os.path.join(sys.path[0], "Face_Detection"))
from user_retrieve import *

class FacialRecognizer:
    def __init__(this):
        this.recognizer = cv2.face.LBPHFaceRecognizer_create()
        this.recognizer.read('Facial_Recognition\\trainer.yml')
        this.font = cv2.FONT_HERSHEY_SIMPLEX
        this.userDB = UserData()
        this.userDB.loadUsernames()
        this.users = this.userDB.getUsernamesList()
        this.currentUser = this.userDB.getCurrentUser()
    
    def predictUser(this,x,y,w,h,img,grayImg):
        # use the recognizer to predict which user is shown
        id, confidence = this.recognizer.predict(grayImg[y:y+h,x:x+w])
        if (confidence < 100):
            id = this.users[id]
        else:
            id = "unknown"
        cv2.putText(img, str(id), (x+5,y-60), this.font, 1, (255,255,255), 2)

