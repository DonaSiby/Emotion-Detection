'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#select face from the input frame
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#predict the emotion
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)
#if 0 passed as argument then it will use the default camera or webcam


#infinity loop to capture the video
#until u dont stop excecution it will keep on capturing the video and predict the emotion
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #convert the frame into gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray scale image into face classifier this detect multiscale method will detect the face from the image frame and data will be stored in 'faces' variable
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #It return 4 values x,y,w,h these are coordinates of the face

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #draw a blue rectangle around the face
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)#resize the image to 48x48 because the model  which we trained is trained is on mobile architecture and in mobilenet architecture we give the input image in this dimensions only


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)#convert the image into array
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class
  
            #line  52 to 58 will predict the emotion
            preds = classifier.predict(roi)[0]#this will predict the probability of each and every class
            print("\nprediction = ",preds) #return an array of probabilities of each class
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax()) #return the index of the class which has maximum probability [preds.argmax() will return the index of the class which has maximum probability]
            print("\nlabel = ",label) #assign the label
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)#put text on the frame or face (ys this is the face and its emotion)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)#show output of the emotion detection program
    if cv2.waitKey(1) & 0xFF == ord('q'):#how to close program if u press 'q' then it will close the program
        break

cap.release()
cv2.destroyAllWindows()


#1.DETECT FACE
#2.DRAW RECTANGLE AROUND THE FACE
#3.PREDICT THE EMOTION
#4.PUT TEXT/LABEL ON THE FACE























