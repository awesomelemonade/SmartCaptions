import subprocess
import math
from time import time
from pathlib import Path
import glob
from collections import namedtuple
import pickle

outputDirectory = "data/test/"
videoURL = "https://www.youtube.com/watch?v=Ye8mB6VsUHw"

debugOut = subprocess.DEVNULL
debugErr = subprocess.DEVNULL

downloadCommand = ["youtube-dl", "-f", "mp4", "-o", outputDirectory + "%(title)s.%(ext)s", "--all-subs",
                        "--sub-format", "vtt", "--sub-lang", "en", videoURL]

startTime = time()
subprocess.run(downloadCommand, stdout=debugOut, stderr=debugErr)
videoPath = glob.glob(outputDirectory + "*.mp4")[0]
captionsPath = glob.glob(outputDirectory + "*.vtt")[0]
print("Downloaded {}".format(videoPath[len(outputDirectory):]))
print("Downloaded {}".format(captionsPath[len(outputDirectory):]))
print("Download took {}ms".format(math.floor((time() - startTime) * 1000)))

# Create frames directory
framesDirectory = outputDirectory + "frames/"
Path(framesDirectory).mkdir(parents=True, exist_ok=True)

fps = 30
convertCommand = ["ffmpeg", "-i", videoPath, "-vf", "fps=fps=" + str(fps),
                    framesDirectory + "frame%d.jpg"]
                    
startTime = time()
subprocess.run(convertCommand, stdout=debugOut, stderr=debugErr)
framePaths = glob.glob(framesDirectory + "*.jpg")
print("Extracted {} images at {} fps in {}ms".format(len(framePaths), fps, math.floor((time() - startTime) * 1000)))

startTime = time()
# Process .vtt captions file
with open(captionsPath) as f:
    captionsLines = [x.rstrip() for x in f]

separators = [i for i, x in enumerate(captionsLines) if len(x) == 0]
split = [captionsLines[separators[i] + 1:separators[i + 1]] for i in range(len(separators) - 1)]

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])

# parses to seconds
def parseTime(s):
    hours, minutes, seconds = s.strip().split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

# get startTime, endTime in seconds
times = [[parseTime(s) for s in lines[0].split("-->")] for lines in split]
# convert to frame numbers
times = [(math.floor(startTime * fps), math.ceil(endTime * fps)) for startTime, endTime in times]
# parse into caption tuples
captions = [Caption(None, ' '.join(lines[1:]), startTime, endTime, None) for (startTime, endTime), lines in zip(times, split)]

print("Parsed {} captions in {} ms".format(len(captions), math.floor((time() - startTime) * 1000)))

startTime = time()

captionsPicklePath = outputDirectory + "captions.pkl"
with open(captionsPicklePath, 'wb') as f:
    pickle.dump(captions, f)

print("Pickled {} captions in {} ms".format(len(captions), math.floor((time() - startTime) * 1000)))


