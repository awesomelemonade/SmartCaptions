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

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])
PrioritizedCaption = namedtuple('PrioritizedCaption', ['time', 'counter', 'caption'])

test = Caption('Bob', 'testing123', 123, 321, None)

framesDir = "./data/friends/frames/"
framePaths = glob.glob(framesDir + "*.jpeg")
print("Reading {} files".format(len(framePaths)), flush=True)

framePaths.sort()

captionsPath = "./data/friends/captions.pkl"
objectsPath = './data/friends/objects.pkl"

with open(captionsPath, 'rb') as captionsFile:
    allCaptions = pickle.load(captionsFile) # list of (character, message, startTime, endTime[, comments])
with open(objectsPath, 'rb') as objectsFile:
    allObjects = pickle.load(objectsFile) # list of {"Character": "N x 2 numpy array"} dictionaries

# Priority Queues to store captions
currentCaptions = [] # heap sorted by endTime
futureCaptions = [] # heap sorted by startTime

for i, caption in enumerate(allCaptions):
    futureCaptions.append(PrioritizedCaption(caption.startTime, i, caption))

heapq.heapify(futureCaptions)

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
    regionsMap = allObjects[i]
    # apply captions
    for _, _, caption in currentCaptions:
        if caption.character in regionsMap:
            # apply w/ objection tracking
            region = regionsMap[caption.character]
        else:
            # apply w/o object tracking

TextRenderer.renderCaption(frame, (100, 100, 100, 100), "sample text")

plt.imshow(frame)
plt.show()
