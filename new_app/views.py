# Create your views here.
from django.http import HttpResponse

from django.shortcuts import render
#from .models import predict

import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2 
import os
import math
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
#import imutils
import pickle
import time
import cv2
import os
import dlib
from scipy.spatial import distance as dist
x = 0




def home(request):
	return render(request,'input.html') 


def output(request):

    import cv2
    from tensorflow.keras.preprocessing.image import img_to_array
    import os
    import numpy as np
    from tensorflow.keras.models import model_from_json

    root_dir = os.getcwd()
    # Load Face Detection Model
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    # Load Anti-Spoofing Model graph
    json_file = open('antispoofing_models/antispoofing_model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load antispoofing model weights 
    model.load_weights('antispoofing_models/antispoofing_model.h5')
    print("Model loaded from disk")
    # video.open("http://192.168.1.101:8080/video")
    # vs = VideoStream(src=0).start()
    # time.sleep(2.0)

    video = cv2.VideoCapture(0)
    while True:
        try:
            ret,frame = video.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                # resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(resized_face)[0]
                print(preds)
                if preds> 0.5:
                    label = 'spoof'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
                else:
                    label = 'real'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except Exception as e:
            pass
    video.release()
    return render(request,'output.html',{'out':" "})
		
