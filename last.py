import cv2
import numpy as np 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
lip_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

video_cap = cv2.VideoCapture(0)
 
emoji = cv2.imread('emoji.png', 0)
emoji_eye = cv2.imread('eye.png', 0)
emoji_lip = cv2.imread('lip.png', 0)

user_input = input("Enter 1 or 2 or 3 or 4 as you like: \n 1.emoji\n 2.lip & eyes\n 3.blured\n 4.flip\n ")


if user_input == "1":
    while True:

        ret, frame = video_cap.read()
        if ret == False:
            break

        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fliped = cv2.flip(frame_g, 1)
        new = np.concatenate((frame_g , fliped) , 1)
        
#         faces = face_detector.detectMultiScale(frame_g, 1.3)
#         for face in faces:
#             x, y, w, h = face

#             emoji_re = cv2.resize(emoji, (h, w))
#             frame_g[y:y+h, x:x+w] = emoji_re


        cv2.imshow('Output', frame_g)
        cv2.waitKey(1) 

elif user_input == "2":
    while True:

        ret, frame = video_cap.read()
        if ret == False:
            break

        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

        eyes = eye_detector.detectMultiScale(frame_g, 1.3 , minNeighbors= 10 )
    
        for eye in eyes:
            x, y, w, h = eye
            eye_re = cv2.resize(emoji_eye, (w, h))
            frame_g[y:y+h, x:x+w] = eye_re



        lips = lip_detector.detectMultiScale(frame_g, 2.5 , minNeighbors= 15 )
        
        for lip in lips:
            x, y, w, h = lip
            lip_re = cv2.resize(emoji_lip, (w, h))
            frame_g[y:y+h, x:x+w] = lip_re

        cv2.imshow('Output', frame_g)
        cv2.waitKey(1) 

elif user_input == "3":
    while True:

        ret, frame = video_cap.read()
        if ret == False:
            break

        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        faces = face_detector.detectMultiScale(frame_g, 1.3)
        for face in faces:
            x, y, w, h = face

            inverted_frame = 255 - frame_g[y:y+h, x:x+w]
            blured_inverted_frame = cv2.GaussianBlur(inverted_frame , (31,31), 0)
            frame_g[y:y+h, x:x+w] = inverted_frame

        cv2.imshow('Output', frame_g)
        cv2.waitKey(1)

elif user_input == "4":
    while True:

        ret, frame = video_cap.read()
        if ret == False:
            break

        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        faces = face_detector.detectMultiScale(frame_g, 1.3)
        for face in faces:
            x, y, w, h = face

            fliped = cv2.flip(frame_g[y:y+h , x:x+w], 1)
            new = np.concatenate((frame_g[y:y+h, x:x+w] , fliped) , 1)

        cv2.imshow('Output', new)
        cv2.waitKey(1)
