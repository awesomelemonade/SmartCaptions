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
import html
from time import time
import math
import subprocess
import imageio
from RankBoundingBoxes import rankBoxes

directory = "./data/peep/"


Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])
PrioritizedCaption = namedtuple('PrioritizedCaption', ['time', 'counter', 'caption'])

framesDir = directory + "frames/"
framePaths = glob.glob(framesDir + "*.jpg")
print("Reading {} files".format(len(framePaths)), flush=True)

framePaths.sort(key=lambda s: int(s.split("\\")[-1][len("frame"):-len(".jpg")]))

captionsPath = directory + "captions.pkl"
objectsPath = directory + "objects.pkl"

with open(captionsPath, 'rb') as captionsFile:
    targetFps = pickle.load(captionsFile)
    allCaptions = pickle.load(captionsFile) # list of (character, message, startTime, endTime[, comments])
with open(objectsPath, 'rb') as objectsFile:
    allObjects = pickle.load(objectsFile)
    objectsByFrameNumber = pickle.load(objectsFile)
    objects = {frameNumber: boxes[0] for frameNumber, boxes in objectsByFrameNumber.items() if boxes}
    #objects = pickle.load(objectsFile) # Currently: {index: BoundingBox} # list of {"Character": "N x 2 numpy array"} dictionaries

# Priority Queues to store captions
currentCaptions = [] # heap sorted by endTime
futureCaptions = [] # heap sorted by startTime

for i, caption in enumerate(allCaptions):
    futureCaptions.append(PrioritizedCaption(caption.startTime, i, caption))

heapq.heapify(futureCaptions)

videoPath = glob.glob(directory + "*.mp4")[0]
outputPathNoAudio = 'out/videoOutput_noAudio.mp4'
outputPathAudio = 'out/videoOutput.mp4'

video = None

for i, path in enumerate(framePaths):
    # update priority queues
    while futureCaptions and futureCaptions[0].time <= i:
        _, count, caption = heapq.heappop(futureCaptions)
        heapq.heappush(currentCaptions, PrioritizedCaption(caption.endTime, count, caption))
    while currentCaptions and currentCaptions[0].time < i:
        heapq.heappop(currentCaptions)
    # read frame
    frame = imageio.imread(path)
    videoWidth, videoHeight, _ = frame.shape
    if video == None:
        video = cv2.VideoWriter(outputPathNoAudio, cv2.VideoWriter_fourcc(*'mp4v'), targetFps, (videoHeight, videoWidth))
    # get regions map by ID
    #regionsMap = allObjects[i]
    regionsMap = {}
    # apply captions
    for j, (_, _, caption) in enumerate(currentCaptions):
        if caption.character in regionsMap:
            # apply w/ objection tracking
            region = regionsMap[caption.character]
        else:
            captionWidth, captionHeight = TextRenderer.getCaptionSize(caption.message)
            if i in objects:
                out = rankBoxes(frame, [captionWidth, captionHeight], objects[i], 1, True)
                # out is empty if and only if there is no associated caption with a frame
                if len(out) != 0:
                    x = out[0][1][1]
                    y = out[0][1][0]
                    TextRenderer.renderCaption(frame, (x, y, captionWidth, captionHeight), caption.message)
            else:
                # apply w/o object tracking
                out = rankBoxes(frame, [captionWidth, captionHeight], None, 1, False)
                # out is empty if and only if there is no associated caption with a frame
                if len(out) != 0:
                    x = out[0][1][1]
                    y = out[0][1][0]
                    TextRenderer.renderCaption(frame, (x, y, captionWidth, captionHeight), caption.message)
    bgrFrame = frame[..., ::-1]
    cv2.imshow("Frame", bgrFrame)
    video.write(bgrFrame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()

combineCommand = ["ffmpeg", "-i", outputPathNoAudio, "-i", videoPath, "-c", "copy",
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", outputPathAudio]

startTime = time()
subprocess.run(combineCommand)
print("Combined video and audio in {}ms".format(math.floor((time() - startTime) * 1000)))

