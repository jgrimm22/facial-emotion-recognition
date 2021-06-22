#!/usr/bin/env python
# coding: utf-8

# # Selbsttrainiertes Modell per Webcam testen

# In[1]:


import os
import cv2
import numpy as np
from keras.models import model_from_json

from keras.preprocessing import image

import sys
import argparse
import re


# Funktion für Face Detections, Emotionsklassifizierung und Darstellung

# In[2]:


def detect_classify_display(frame, model):
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'sad', 'surprise', 'neutral']
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = frame_gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        predictions = model.predict(face)
        pred = np.argmax(predictions)
        prob = predictions[0, pred] * 100
        frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
        cv2.putText(frame, emotions[pred] + f'   {prob:.2f}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.imshow('Emotion recognition', frame)


# ## Modell laden
# Laden des vortrainierten Modells und der Gewichte

# In[3]:


json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("fer.h5")
print("Loaded model from disk")


# ## Face Detection
# Für die Face Detection wird ein über OpenCV verfügbarer Haar-feature-basierter Cascade Classifier verwendet.

# In[4]:


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')


# ## FER über die Webcam
# get the frame from webcam and detect

# In[5]:


cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    detect_classify_display(frame, model)
    c = cv2.waitKey(1)
    if cv2.waitKey(0) & 0xff == ord('q'): # press q to exit
        cv2.destroyAllWindows()
        break
        


# ## Fazit 
# 
# <img src="../images/test/test.gif" width="500" align="center">
# 
# * Das Netzwerk tendiert stark zur neutralen Klasse, auch bei ausgeprägter Mimik 
# * Anger und Disgust sind am schwersten nachzustellen, sodass das Modell diese erkennt
# * Generell reagiert das Netz ausschließlich auf sehr ausgeprägte Mimik, was vermutlich an dem FER2013 Dataset liegt
# * Eine relativ grobe Emotionserkennung wird bereits durch dieses recht einfache Modell bewältigt, aber eine feinere Emotionserkennung ist auf Basis dieses Datensatzes und des Netzes eher schwer zu realisieren

# In[ ]:




