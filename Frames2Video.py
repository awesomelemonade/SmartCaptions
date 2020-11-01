import numpy as np
import imageio

# ~~~~requires imageio-ffmpeg~~~~
# given a list of frames (numpy arrays), specifically an array of size ((wxhx3)xn) where n is the number of frames,
# convert the sequence of frames into a video
# fps here is 5 to match the sample rate of the initial video -> series of frames
def frames2video(frames):
    imageio.mimwrite('videoOutput.mp4', frames, fps=5)