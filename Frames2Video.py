import numpy as np
import imageio

from os import listdir
from os.path import isdir

# ~~~~requires imageio-ffmpeg~~~~
# given a list of frames (numpy arrays), specifically an array of size ((wxhx3)xn) where n is the number of frames,
# convert the sequence of frames into a video
# fps here is 5 to match the sample rate of the initial video -> series of frames
def frames2video(frames):
    imageio.mimwrite('videoOutput.mp4', frames, fps=5)

test_frames = []

videoId = "M7FIvfx5J10"
frame_rate = 5
i = 1
while True:
    try:
        test_frames.append(imageio.imread(f"./frames/frame{i}_{frame_rate}fps_{videoId}.jpg"))
        i += 1
    except Exception:
        break

frames2video(test_frames)