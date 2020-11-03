import cv2
import numpy as np
from pydub import AudioSegment # https://github.com/jiaaro/pydub
from pydub.playback import play
from pydub.playback import _play_with_simpleaudio
import time

CONTROL_PLAY_PAUSE = ord('k')
CONTROL_PLAY_PAUSE_ALIAS = ord(' ')
CONTROL_PREV = ord('j')
CONTROL_FORWARD = ord('l')
CONTROL_FAST_PREV = ord('h')
CONTROL_FAST_FORWARD = ord(';')
CONTROL_SELECT_START = ord('s')
CONTROL_SELECT_STOP = ord('d')

window = 'SelectROIs'

cv2.namedWindow(window)

videoPath = 'data/bfdi1a/BFDI 1a - Take the Plunge.mp4'
capture = cv2.VideoCapture(videoPath)
audio = AudioSegment.from_file(videoPath)

frameRate = capture.get(cv2.CAP_PROP_FPS)

totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

timeSlider = 'Time'
cv2.createTrackbar(timeSlider, window, 0, totalFrames, lambda x: None)

isPlaying = False

playTime = None

selectStartFrame = None
selectStartBox = None
selectStopFrame = None
audioPlayback = None

def startVideoAtFrame(currentFrame):
    global isPlaying, audioPlayback, playTime
    isPlaying = True
    # play audio
    sliced = audio[int(currentFrame / totalFrames * len(audio)):]
    beforeTime = time.time()
    audioPlayback = _play_with_simpleaudio(sliced)
    playTime = beforeTime - currentFrame / frameRate
def stopVideo():
    global audioPlayback, isPlaying
    isPlaying = False
    if audioPlayback != None:
        audioPlayback.stop()
    audioPlayback = None

while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE):
    currentFrame = cv2.getTrackbarPos(timeSlider, window)
    
    if isPlaying:
        timePassed = time.time() - playTime
        cv2.setTrackbarPos(timeSlider, window, round(timePassed * frameRate))
    
    currentFrame = cv2.getTrackbarPos(timeSlider, window)
    capture.set(cv2.CAP_PROP_POS_FRAMES, currentFrame - 1)
    
    success, frame = capture.read()
    
    cv2.imshow(window, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if currentFrame == totalFrames:
        stopVideo()
    if key == CONTROL_PLAY_PAUSE or key == CONTROL_PLAY_PAUSE_ALIAS:
        if isPlaying:
            stopVideo()
        else:
            startVideoAtFrame(currentFrame)
    if key == CONTROL_SELECT_START:
        stopVideo()
        selectStartFrame = currentFrame
        selectStartBox = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, selectStartBox)
        startVideoAtFrame(currentFrame)
    if key == CONTROL_SELECT_STOP:
        if selectStartBox != None:
            stopVideo()
            selectStopFrame = currentFrame
            # process csrt
            frameHeight, frameWidth, _ = frame.shape
            numFrames = selectStopFrame - selectStartFrame + 1
            progress = np.zeros((20, frameWidth, 3), dtype=np.uint8)
            for i in range(numFrames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, selectStartFrame + i - 1)
                success, frame = capture.read()
                success, box = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                progress[:, :round(i / numFrames * frameWidth), 1] = 255
                cv2.imshow(window, np.vstack((frame, progress)))
                cv2.waitKey(1)
            cv2.setTrackbarPos(timeSlider, window, currentFrame + 1)
            selectStartFrame = None
            selectStartBox = None
    if key == CONTROL_FAST_PREV:
        cv2.setTrackbarPos(timeSlider, window, currentFrame - 5)
    if key == CONTROL_FAST_FORWARD:
        cv2.setTrackbarPos(timeSlider, window, currentFrame + 5)
    if key == CONTROL_PREV:
        cv2.setTrackbarPos(timeSlider, window, currentFrame - 1)
    if key == CONTROL_FORWARD:
        cv2.setTrackbarPos(timeSlider, window, currentFrame + 1)
    if key == ord('q'):
        break


cv2.destroyAllWindows()



