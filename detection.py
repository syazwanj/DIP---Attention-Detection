import numpy as np
import cv2
import time
from keras.models import load_model
# https://www.youtube.com/watch?v=mPCZLOVTEc4

def draw_defiance_box(frame, x, y):
    cv2.rectangle(frame, (x//2, y-30), (x//2+30, y), RED, 15)
    cv2.putText(
        frame, "PAY ATTENTION", (x//2, y-15), FONT, 1,
        WHITE, 1, cv2.LINE_AA
        )

# Set paths
EYE_MODEL_PATH = './drowsiness_files/models/cnncat2.h5'

# Configs
FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
BLUE = (255,0,0)
GREEN = (0,255,0)
GREEN_LEFT = (0,255,0)
GREEN_RIGHT = (255,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
THICC = 5
NAUGHTY_BOI_THRESHOLD = 3

# Import detection files
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

left_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'
    )

right_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml'
    )

mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
    )

# Load in trained model(s)
eye_model = load_model(EYE_MODEL_PATH)

# Initialise timer
time_elapsed = 0
start_time = 0


cap = cv2.VideoCapture(0)
count = 0

# Take images as input with webcam until user presses 'q'
while True:
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]

    # OpenCV algo for obj detection takes gray images for input.
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
        # Draw rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), RED, THICC)

        # Regions of intersection for each face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect left and right eyes
        left_eye = left_eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        left_label = 0
        right_label = 0

        for (ex, ey, ew, eh) in left_eye:
            eye_box = roi_gray[ey:ey+eh, ex:ex+ew]
            count+=1
            eye_box = cv2.resize(eye_box, (24,24)) # Model is trained on 24x24 img
            eye_box = eye_box/255 # Normalisation
            eye_box = eye_box.reshape(24, 24, -1) # Flatten
            eye_box = np.expand_dims(eye_box, axis = 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN_LEFT, THICC)
            left_pred = eye_model.predict(eye_box)
            if left_pred[0][0] <= 0.8:
                left_label = 1

        for (ex, ey, ew, eh) in right_eye:
            eye_box = roi_gray[ey:ey+eh, ex:ex+ew]
            count+=1
            eye_box = cv2.resize(eye_box, (24,24)) # Model is trained on 24x24 img
            eye_box = eye_box/255 # Normalisation
            eye_box = eye_box.reshape(24, 24, -1) # Flatten
            eye_box = np.expand_dims(eye_box, axis = 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN_RIGHT, THICC)
            right_pred = eye_model.predict(eye_box)
            if right_pred[0][0] <= 0.8:
                right_label = 1

        text_label = 'Closed'
        if left_label and right_label:
            text_label = 'Open'
        else:
            if start_time == 0:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time >= NAUGHTY_BOI_THRESHOLD:
                draw_defiance_box(frame, frame_width, frame_height)

        cv2.putText(
            frame, text_label, (10, frame_height-20), FONT, 1,
            WHITE, 1, cv2.LINE_AA
            )

        # Mouth detection
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 1.3, 13)
        if isinstance(mouth_rects, np.ndarray):
            if mouth_rects.size != 0:
                # mouth_rects = mouth_rects[0]
                mx, my, mw, mh = mouth_rects[0]
            # for mx, my, mw, mh in mouth_rects:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), BLUE, THICC)

            
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break




