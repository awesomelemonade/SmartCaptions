import subprocess
import math
from time import time
from pathlib import Path
import glob
from collections import namedtuple
import pickle
import html

### Arguments ###

outputDirectory = "data/peep/"
videoURL = "https://www.youtube.com/watch?v=Tz0KPOXlCbM"
auto = False

debugOut = subprocess.DEVNULL
debugErr = subprocess.DEVNULL

#debugOut, debugErr = None, None

### Script Below ###


downloadCommand = ["youtube-dl", "-f", "mp4", "-o", outputDirectory + "%(title)s.%(ext)s", "--write-auto-sub" if auto else "--write-sub",
                        "--sub-format", "ttml", "--sub-lang", "en", videoURL]

startTime = time()
subprocess.run(downloadCommand, stdout=debugOut, stderr=debugErr)
videoPath = glob.glob(outputDirectory + "*.mp4")[0]
captionsPath = glob.glob(outputDirectory + "*.ttml")[0]
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
    captionsLines = [x.rstrip() for x in f if x.startswith("<p")]

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])

# parses to seconds
def parseTime(s):
    hours, minutes, seconds = s.strip().split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

# get startTime, endTime in seconds
times = [[parseTime(s) for s in line.split("\"")[1:5:2]] for line in captionsLines]
# convert to frame numbers
times = [(math.floor(startTime * fps), math.ceil(endTime * fps)) for startTime, endTime in times]
# parse messages
messages = [line[line.index(">") + 1:] for line in captionsLines]
messages = [line[:line.index("<")] for line in messages]
messages = [html.unescape(line) for line in messages]
# parse into caption tuples
captions = [Caption(None, line, startTime, endTime, None) for (startTime, endTime), line in zip(times, messages)]

print("Parsed {} captions in {} ms".format(len(captions), math.floor((time() - startTime) * 1000)))

startTime = time()

captionsPicklePath = outputDirectory + "captions.pkl"
with open(captionsPicklePath, 'wb') as f:
    pickle.dump(fps, f)
    pickle.dump(captions, f)

print("Pickled {} captions in {} ms".format(len(captions), math.floor((time() - startTime) * 1000)))


