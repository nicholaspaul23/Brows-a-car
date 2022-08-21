import cv2
from PIL import Image
import numpy as np
import os
import sys

from user_retrieve import *
sys.path.append(os.path.join(sys.path[0], "Face_Detection"))
from Face_Detection.capture import *

def processFaceData():
    # return a list of detected faces and paired ids for each image in our Face_Data
    # O(n**2) optimize later
    face_classifier = cv2.CascadeClassifier(os.path.join(sys.path[0], "Face_Detection\\faceCascade.xml")) # sys path 0 being the root directory the script is run -> src
    path = os.path.join(sys.path[0], 'Facial_Recognition\\Face_Data')
    imagePaths = [os.path.join(path,file) for file in os.listdir(path)]
    faceSamples = []
    ids = []
    userDB = UserData()
    userDB.loadUsernames()
    for image in imagePaths:
        PIL_img = Image.open(image).convert('L') # use to convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8') # 8 bit for values 0-255
        id = os.path.split(image)[-1].split(".")[1] # parse user id from filename
        id = int(userDB.getUsernamesList().index(id))
        faces = face_classifier.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

def trainFacialRecognizer(faces,ids):
    # create and train a model to recognize faces using the recognizer Local binary patterns histograms recognizer included in OpenCV
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write('Facial_Recognition\\trainer.yml')