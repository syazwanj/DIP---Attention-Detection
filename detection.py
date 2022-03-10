import numpy as np
import cv2
# https://www.youtube.com/watch?v=mPCZLOVTEc4

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
    )

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    '''
    1.3 = scaleFactor: specifies how much the image size is reduced at
    each image scale. Smaller value: higher accuracy but slower performance.
    5 = minNeighbours: Specifies how many neighbours each candidate 
    rectangle should have to retain it. Affects the quality of affected
    faces.
    '''
    for (x, y, w, h) in faces:
        gray_2 = gray.copy()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_2 = gray_2[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 13)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

        # Mouth detection
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray_2, 1.3, 13)
        if isinstance(mouth_rects, np.ndarray):
            if mouth_rects.size != 0:
                # mouth_rects = mouth_rects[0]
                mx, my, mw, mh = mouth_rects[0]
            # for mx, my, mw, mh in mouth_rects:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 5)

            
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break



