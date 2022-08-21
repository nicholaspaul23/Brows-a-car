from asyncio.windows_events import NULL
from msilib.schema import Class
import cv2
import os
import sys

# This class provides methods for video capture and facial detection using Haar Cascade classifier

class FaceDetector:
    def __init__(this, id=NULL):
        this.face_id = id
        # initialize path for storage space for captured images (notice this works primarily on NTFS windows)
        this.face_data_path = os.path.join(sys.path[0], "Facial_Recognition\\Face_Data")
    
    def __del__(this):
        this.cam.release()
        cv2.destroyAllWindows()
    
    # create a classifier model
    face_classifier = cv2.CascadeClassifier(os.path.join(sys.path[0], "Face_Detection\\faceCascade.xml")) # sys path 0 being the root directory the script is run -> src

    # count for captures taken
    captureCount = 0

    def setUserId(this, userId):
        # for a new user create a face id
        this.face_id = userId
    
    def returnUserId(this):
        # return current user
        return this.face_id
    
    def capture(this):
        # check if user is new
        if (this.face_id == NULL):
            this.initializeUser()
        # start the camera module
        this.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # To resolve bug in MSMF backend on Windows set param to Direct Show (https://stackoverflow.com/questions/60007427/cv2-warn0-global-cap-msmf-cpp-674-sourcereadercbsourcereadercb-termina)
        this.cam.set(3, 640)
        this.cam.set(4, 480)
    
    def detectFaces(this):
        # use cv2 cam module to find faces, based on amount of faces found the list of bounding boxes (rectangle coordinates) are returne in var "faces"
        ret, img = this.cam.read()
        img = cv2.flip(img, 1) # flip img
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FaceDetector.face_classifier.detectMultiScale(grayImg, 1.3, 5)
        return img, grayImg, faces

    def processFaces(this,img,grayImg,faces):
        # capture/save faces found, show bounding box over detected face
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.imwrite(this.face_data_path + "\\User." + str(this.face_id) + '.' + str(this.captureCount) + '.jpg', grayImg[y:y+h,x:x+w])
            this.captureCount = this.captureCount + 1

    def processFaces_Mod(this,img,grayImg,faces,*funcs):
        # iterate through faces found, show bounding box, call callback functions if set
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            # !!! ToDo: Optimize/get rid of "packing" bc this can become O(n^3)
            for func in funcs:
                func(x,y,w,h,img,grayImg)
    
    def processFaces_Mod_Opt(this,img,grayImg,faces,func1=None,func2=None):
        # Note: Optimized version of the above, until a better solution can be found
        # iterate through faces found, show bounding box, call callback function if set
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            if func1 is not None:
                func1(x,y,w,h,img,grayImg)
            if func2 is not None:
                func2(x,y,w,h,img,grayImg)
    
    def processFaces_Mod_Single(this,img,grayImg,faces,func):
        # iterate through faces found, show bounding box, call callback function if set
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            func(x,y,w,h,img,grayImg)
    
    def showImage(this,img):
        # show the camera video feed
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            sys.exit("ESC key hit, ending program...")