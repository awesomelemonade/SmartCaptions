import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import TextRenderer
import pickle
from collections import namedtuple
import heapq
import imageio

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])
PrioritizedCaption = namedtuple('PrioritizedCaption', ['time', 'counter', 'caption'])

directory = "./data/joe_and_lex/"

framesDir = directory + "frames/"
framePaths = glob.glob(framesDir + "*.jpg")
print("Reading {} files".format(len(framePaths)), flush=True)

framePaths.sort(key=lambda s: int(s.split("/")[-1][len("frame"):-len(".jpg")]))

captionsPath = directory + "captions.pkl"
objectsPath = directory + "objects.pkl"

with open(captionsPath, 'rb') as captionsFile:
    targetFps = pickle.load(captionsFile)
    allCaptions = pickle.load(captionsFile) # list of (character, message, startTime, endTime[, comments])
with open(objectsPath, 'rb') as objectsFile:
    objects = pickle.load(objectsFile) # Currently: {index: BoundingBox} # list of {"Character": "N x 2 numpy array"} dictionaries

# Priority Queues to store captions
currentCaptions = [] # heap sorted by endTime
futureCaptions = [] # heap sorted by startTime

for i, caption in enumerate(allCaptions):
    futureCaptions.append(PrioritizedCaption(caption.startTime, i, caption))

heapq.heapify(futureCaptions)


frame_list = []

for i, path in enumerate(framePaths):
    # update priority queues
    while futureCaptions and futureCaptions[0].time <= i:
        _, count, caption = heapq.heappop(futureCaptions)
        heapq.heappush(currentCaptions, PrioritizedCaption(caption.endTime, count, caption))
    while currentCaptions and currentCaptions[0].time < i:
        heapq.heappop(currentCaptions)
    # read frame
    frame = imageio.imread(path)
    # get regions map by ID
    #regionsMap = allObjects[i]
    regionsMap = {}
    # apply captions
    for _, _, caption in currentCaptions:
        if caption.character in regionsMap:
            # apply w/ objection tracking
            region = regionsMap[caption.character]
        else:
            captionWidth, captionHeight = TextRenderer.getCaptionSize(caption.message)
            if i in objects:
                (x, y, width, height), class_name, confidence = objects[i]
                TextRenderer.renderCaption(frame, (x, y, captionWidth, captionHeight), caption.message)
            else:
                # apply w/o object tracking
                TextRenderer.renderCaption(frame, (100, 100, captionWidth, captionHeight), caption.message)
    frame_list.append(frame)
    cv2.imshow("Frame", frame[...,::-1])
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ~~~~requires imageio-ffmpeg~~~~
# given a list of frames (numpy arrays), specifically an array of size ((wxhx3)xn) where n is the number of frames,
# convert the sequence of frames into a video
def frames2video(frames):
    imageio.mimwrite('out/videoOutput.mp4', frames, fps=targetFps)

frames2video(frame_list)

