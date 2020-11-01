#### FROM https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt
# python opencv_object_tracking.py --tracker csrt
# ^^^ will do computer camera
# CSRT is said to be the best but slowest to process, all p fast though
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import pickle
import split_video_into_scenes
import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

vs = cv2.VideoCapture(args["video"])

objects = {}
frameNumber = 0
frameRate = vs.get(cv2.CAP_PROP_FPS)
print("Frame Rate: {}".format(frameRate))

# initialize the FPS throughput estimator
fps = None
scenes = split_video_into_scenes.find_scenes(args["video"])
scene_count = 0
# loop over frames from the video stream
while True:
	# grab the current frame
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	
	(H, W) = frame.shape[:2]

	key = cv2.waitKey(1) & 0xFF

	if (scene_count < len(scenes) and frameNumber == scenes[scene_count][0].frame_num) or key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker = cv2.TrackerCSRT_create()
		tracker.init(frame, initBB)
		fps = FPS().start()
		scene_count += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
	frameNumber += 1

	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)

		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			objects[frameNumber] = (x, y, w, h)
			cv2.rectangle(frame, (x, y), (x + w, y + h),
						  (0, 255, 0), 2)

		# update the FPS counter
		fps.update()
		fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
vs.release()
print(frameNumber)
print(frameRate)
# close all windows
cv2.destroyAllWindows()


# Resample

resampled = {}
totalTime = frameRate * frameNumber
targetFps = 30

for i in range(round(totalTime * targetFps)):
    frameIndex = round(i / targetFps * frameRate)
    if frameIndex in objects:
        resampled[i] = objects[frameIndex]

outputDirectory = "data/test/"
objectsPicklePath = outputDirectory + "objects.pkl"
with open(objectsPicklePath, 'wb') as f:
    pickle.dump(resampled, f)





