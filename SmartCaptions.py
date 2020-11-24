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
from RankBoundingBoxes import rankBoxesFast, getEnergyMatrix
from pydub import AudioSegment

directory = "./data/peep/"

videoPath = glob.glob(directory + "*.mp4")[0]
audio = AudioSegment.from_file(videoPath)

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])
PrioritizedCaption = namedtuple('PrioritizedCaption', ['time', 'counter', 'caption'])

framesDir = directory + "frames/"
framePaths = glob.glob(framesDir + "*.jpg")
print("Reading {} files".format(len(framePaths)), flush=True)

framePaths.sort(key=lambda s: int(s.split(os.path.sep)[-1][len("frame"):-len(".jpg")]))
totalFrames = len(framePaths)

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

captionVolumes = [audio[round(startTime / totalFrames * len(audio)):round(endTime / totalFrames * len(audio))].max for _, _, startTime, endTime, _ in allCaptions]
captionVolumes = {caption: volume for caption, volume in zip(allCaptions, captionVolumes)}

# Priority Queues to store captions
currentCaptions = [] # heap sorted by endTime
futureCaptions = [] # heap sorted by startTime

for i, caption in enumerate(allCaptions):
    futureCaptions.append(PrioritizedCaption(caption.startTime, i, caption))

heapq.heapify(futureCaptions)

videoPath = glob.glob(directory + "*.mp4")[0]
outputPathNoAudio = 'out/videoOutput_noAudio.mp4'
outputPathEnergy = 'out/videoOutput_energy.mp4'
outputPathAudio = 'out/videoOutput.mp4'

video = None
energyVideo = None

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
    # apply captions
    for j, (_, _, caption) in enumerate(currentCaptions):
        scale = max(0.25, captionVolumes[caption] / 16000)
        captionWidth, captionHeight = TextRenderer.getCaptionSize(caption.message, scale)
        if captionWidth >= videoWidth:
            scale = 0.6
            captionWidth, captionHeight = TextRenderer.getCaptionSize(caption.message, scale)
        
        if j == 0:
            energyMatrix = getEnergyMatrix(frame, [captionHeight, captionWidth], captionId=id(caption), objBox=objects[i] if i in objects else None)
            if energyVideo == None:
                energyWidth, energyHeight = energyMatrix.shape
                energyVideo = cv2.VideoWriter(outputPathEnergy, cv2.VideoWriter_fourcc(*'mp4v'), targetFps, (energyHeight, energyWidth))
            energyMatrix[energyMatrix == np.inf] = 0
            energyMatrix = energyMatrix / 100
            
            energyMatrix[energyMatrix < 0] = 0
            energyMatrix[energyMatrix > 255] = 255
            
            energyMatrix = np.repeat(energyMatrix[..., np.newaxis].astype(np.uint8), 3, axis=2)
        
        out = rankBoxesFast(frame, [captionHeight, captionWidth], captionId=id(caption), objBox=objects[i] if i in objects else None)
        # out is empty if and only if there is no associated caption with a frame
        if len(out) != 0:
            x, y = out[0]
            TextRenderer.renderCaption(frame, (x, y, captionWidth, captionHeight), caption.message, scale)
    if energyVideo is not None:
        energyVideo.write(energyMatrix)
    bgrFrame = frame[..., ::-1]
    cv2.imshow("Frame", bgrFrame)
    video.write(bgrFrame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()
energyVideo.release()

combineCommand = ["ffmpeg", "-i", outputPathNoAudio, "-i", videoPath, "-c", "copy",
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", outputPathAudio]

startTime = time()
subprocess.run(combineCommand)
print("Combined video and audio in {}ms".format(math.floor((time() - startTime) * 1000)))

