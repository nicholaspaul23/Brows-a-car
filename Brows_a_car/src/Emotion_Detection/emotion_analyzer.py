from ast import Lambda
import cv2
import time
import math
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
        this.emo_scale_map = {
            1: 0, # Disgusted
            0: 1, # Angry
            2: 2, # Fearful
            5: 3, # Sad
            4: 4, # Neutral
            3: 5, # Happy
            6: 6  # Surprised
        }
        this.pics_dict = {
            0: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Honda.jpg"),
                "data" : [20,2015,29,1,2],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 1
            },
            1: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\BMW.jpg"),
                "data" : [0,2011,0,0,0],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 0
            },
            2: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Maybach.jpg"),
                "data" : [13,2010,22,2,2],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 4
            },
            3: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Mercedes.jpg"),
                "data" : [3,1990,3,2,0],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 2
            },
            4: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Toyota.jpg"),
                "data" : [11,2016,7,1,6],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 1
            },
            5: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Volvo.jpg"),
                "data" : [6,1995,24,1,2],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 5
            },
            6: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Saab.jpg"),
                "data" : [16,2012,25,0,1],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 6
            },
            7: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\Audi.jpg"),
                "data" : [1,2016,3,0,2],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 0
            },
            8: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\GMC.jpg"),
                "data" : [17,2015,20,1,6],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 1
            },
            9: {
                "path" : os.path.join(rootPath, "Emotion_Detection\\car_pics\\BMW2.jpg"),
                "data" : [0,2015,30,2,2],
                "emotionsRecorded" : [],
                "avgEmotion": 0,
                "cluster": 7
            }
        }
        this.pics_count = len(this.pics_dict) - 1
        this.userDB = UserData()
        this.userDB.loadUsernames()
        this.users = this.userDB.getUsernamesList()
        this.currentUser = this.userDB.getCurrentUser()
        this.facialRec = FacialRecognizer()
        this.cam = FaceDetector(this.currentUser)
        this.emotion_prediction = 4
        this.clusterTallies = []
        this.recCluster = -1
        this.avg = lambda arr : int(round(((sum(arr) / len(arr)) + (max(set(arr), key=arr.count) * 2)) / 3)) # calculate a weighted average, avg(avg + (mode * 2)), giving more weight to the most repeated emotion
  
    def processEmotionResponse(this, withRec = False):
        # capture faces, recognize user (optional), show car pictures, detect emotion response
        this.cam.capture()
        iterations = 0
        picsCount = this.pics_count
        this.showCarPicture(picsCount)
        while iterations <= len(this.pics_dict) * 30:
            img, grayImg, faces = this.cam.detectFaces()
            if iterations % 30 == 0 and iterations >= 30:
                this.pics_dict[picsCount]["avgEmotion"] = this.avg(this.pics_dict[picsCount]["emotionsRecorded"])
                picsCount -= 1
                if picsCount < 0: break
                this.showCarPicture(picsCount)
            if (withRec):
                this.cam.processFaces_Mod_Opt(img,grayImg,faces,this.facialRec.predictUser,this.predictEmotion)
            else:
                this.cam.processFaces_Mod_Single(img,grayImg,faces,this.predictEmotion)
            this.cam.showImage(img,window="Face",x=700,y=30)
            iterations += 1
            this.pics_dict[picsCount]["emotionsRecorded"].append(this.emo_scale_map[this.emotion_prediction])
    
    def showCarPicture(this, count):
        # show car picture
        carImg = cv2.imread(this.pics_dict[count]["path"])
        cv2.imshow('car',carImg)
    
    def predictEmotion(this,x,y,w,h,img,grayImg):
        # predict emotion
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_grayImg = grayImg[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_grayImg, (48, 48)), -1), 0)
        # gather the array of predictions, return index of emotion that had the most entries
        emotion_data = this.FER_MODEL.predict(cropped_img)
        this.emotion_prediction = int(np.argmax(emotion_data))
        cv2.putText(img, this.emo_dict[this.emotion_prediction], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def tallyResults(this):
        # tally the responses for each image, if the response was positive add the cluster number to the winning set
        for key in this.pics_dict:
            # if the average response was happy or surprised
            if this.pics_dict[key]["avgEmotion"] > 4:
                this.clusterTallies.append(this.pics_dict[key]["cluster"])
    
    def returnRecCluster(this)-> int:
        # return the cluster with the most hits
        if (not this.clusterTallies):
            # if there were no results return inconclusive result
            print("Inconclusive analysis, try again")
            return -1
        this.recCluster = max(set(this.clusterTallies), key=this.clusterTallies.count) # The key argument with the count() method compares and returns the number of times each element is present in the data set
        print(f"Here is your recommended Car Cluster based on analysis: {this.recCluster}")
        return this.recCluster
