import cv2
import sys
import pickle
import imutils
import imageio
from itertools import count
import numpy as np
import os

def process_face(img):
    if img.shape != face_shape:
        img = cv2.resize(img, face_shape[:2])
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

if __name__ == '__main__' :

    # tracker_type = "CSRF"
    # tracker = cv2.TrackerCSRT_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_shape = (32,32,3)
    video_name = "joe_and_lex"
    face_names = ["joe", "lex"]
    faces = []
    labels = []
    for label, face_name in enumerate(face_names):
        for i in count(1, 1):
            fname = f"faces/{video_name}/{face_name}{i}.jpg"
            if os.path.isfile(fname):
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                img = process_face(img)
                faces.append(img)
                labels.append(label)
            else:
                break
    face_recognizer.train(faces, np.array(labels))


    # Read video
    video = cv2.VideoCapture("videos/Joe Rogan & Lex Fridman - Are Elon Musk's Fears About AI Realistic.mp4")
    frameRate = video.get(cv2.CAP_PROP_FPS)
    # skip first 10 frames
    for i in range(10):
        ok, frame = video.read()

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    for i in range(10):
        video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Define an initial bounding box
    # bbox = cv2.selectROI(frame)

    # Initialize tracker with first frame and bounding box
    # ok = tracker.init(frame, bbox)
    objects = {}
    for frame_idx in count():
        # Read a new frame
        ok, frame = video.read()

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        save_face = False
        if k == ord("s") : 
            save_face = True
        if k == ord("q") : break

        if not ok:
            break
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)
        rects = face_cascade.detectMultiScale(frame)
        # ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        areas = []
        # Draw bounding box
        if ok and len(rects) > 0:
            # Tracking success
            # p1 = (int(bbox[0]), int(bbox[1]))
            # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            for (x,y,w,h) in rects:
                areas.append(w*h)
            face_roi = rects[np.argmax(areas)]
            x,y,w,h = face_roi
            face = frame[y:y+h, x:x+w]
            face = process_face(face)
            if save_face == True:
                path = "faces/" + video_name + "/" + input("what do we call it?") + ".jpg"
                print("writing face to " + path)
                cv2.imwrite(path, face)
            face_label, confidence = face_recognizer.predict(face)
            face_class = face_names[face_label]
            print(f"face classified as {face_class} with confidence: {confidence}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            objects[frame_idx] = (face_roi, face_class, confidence)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # # Display tracker type on frame
        # cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)


    video.release()

    # close all windows
    cv2.destroyAllWindows()

    # Resample

    resampled = {}
    totalTime = frameRate * (frame_idx+1)
    targetFps = 30

    for i in range(round(totalTime * targetFps)):
        frame_idx = round(i / targetFps * frameRate)
        if frame_idx in objects:
            resampled[i] = objects[frame_idx]

    outputDirectory = "data/test/"
    os.makedirs(outputDirectory, exist_ok=True);
    objectsPicklePath = outputDirectory + "objects.pkl"
    with open(objectsPicklePath, 'wb') as f:
        pickle.dump(resampled, f)