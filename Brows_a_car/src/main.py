import cv2
import os

import sys
rootFile = os.path.realpath(__file__)
dirname = os.path.dirname(rootFile)
sys.path.append(os.path.join(dirname, "Face_Detection"))  # "src/Face_Detection"
sys.path.append(os.path.join(dirname, "Facial_Recognition")) # "scr/Facial_Recognition"
sys.path.append(os.path.join(dirname, "Emotion_Detection")) # "src/Emotion_Detection"
sys.path.append(os.path.join(dirname, "Cluster_Segmentation")) # "src/Cluster_Segmentation"

# Modules
from Face_Detection.capture import *
from Facial_Recognition.user_retrieve import *
from Facial_Recognition.train import processFaceData, trainFacialRecognizer
from Facial_Recognition.recognizer import beginFacialRecognition
from Emotion_Detection.emotion_analyzer import *
from init_user import initUser, initializeUser
from Cluster_Segmentation.cluster_predict import *



# Driver
try:
    userDB = UserData()
    userDB.loadUsernames()
    isNewUser, returnUser = initUser()
    if (returnUser != "" and not userDB.checkUserExist(returnUser)): sys.exit("User not found, try again")
    cam = FaceDetector() if (str(isNewUser) == 'y') else FaceDetector(returnUser)
    cam.setUserId(initializeUser()) if (str(isNewUser) == 'y') else None
    userDB.setCurrentUser(cam.returnUserId())
    cam.capture()
    trainingPhotosToBeTaken = 31 # change here for scalability and more accurate training models
    if(str(isNewUser) == 'y'):
        while(cam.captureCount < trainingPhotosToBeTaken):
            img, grayImg, faces = cam.detectFaces()
            cam.processFaces(img,grayImg,faces)
            cam.showImage(img)
        faces,ids = processFaceData()
        trainFacialRecognizer(faces,ids)
    beginFacialRecognition(userDB.getCurrentUser())
    emoAnalyzer = EmoAnalyzer()
    emoAnalyzer.processEmotionResponse(withRec=True)
    emoAnalyzer.tallyResults()
    cluster = emoAnalyzer.returnRecCluster()
    clusterPred = ClusterPredict()
    clusterPred.suggestThreeCars(cluster, printRes=True)

except KeyboardInterrupt:
    print("Exiting program...")
    pass
