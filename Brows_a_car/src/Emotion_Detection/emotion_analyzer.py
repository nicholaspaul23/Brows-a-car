import cv2
import numpy as np
import os
import sys
from keras.models import load_model

sys.path.append(os.path.join(sys.path[0], "Face_Detection"))
sys.path.append(os.path.join(sys.path[0], "Facial_Recognition"))
sys.path.append(os.path.join(sys.path[0], "FER_CNN"))
from Face_Detection.capture import *
from Facial_Recognition.recognizer_util import *
from Facial_Recognition.user_retrieve import *

class EmoAnalyzer:
    def __init__(this):
        rootPath = os.path.abspath(os.path.join(__file__,"..\.."))
        modelFile = os.path.join(rootPath, "FER_CNN\\fer_model_85.h5")
        this.FER_MODEL = load_model(modelFile)
        this.emo_dict = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"
        }
        this.userDB = UserData()
        this.userDB.loadUsernames()
        this.users = this.userDB.getUsernamesList()
        this.currentUser = this.userDB.getCurrentUser()
        this.facialRec = FacialRecognizer()
        this.cam = FaceDetector(this.currentUser)
  
    def processEmotionResponse(this, withRec = False):
        # capture faces, recognize user (optional), detect emotion response
        this.cam.capture()
        while True:
            img, grayImg, faces = this.cam.detectFaces()
            if (withRec):
                this.cam.processFaces_Mod_Opt(img,grayImg,faces,this.facialRec.predictUser,this.predictEmotion)
            else:
                this.cam.processFaces_Mod_Single(img,grayImg,faces,this.predictEmotion)
            this.cam.showImage(img)

    def predictEmotion(this,x,y,w,h,img,grayImg):
        # predict emotion
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_grayImg = grayImg[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_grayImg, (48, 48)), -1), 0)

        emotion_prediction = this.FER_MODEL.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(img, this.emo_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)