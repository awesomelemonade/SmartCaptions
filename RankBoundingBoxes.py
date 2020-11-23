import numpy as np
from scipy import ndimage
from copy import deepcopy
from shapely.geometry import Polygon, Point
import scipy.io
import os
import matplotlib.pyplot as plt
# if testing with single frames, uncomment the import below and make sure selectRegion is visible
# from selectRegion import roipoly
from matplotlib.patches import Rectangle
import imageio
import math

# convert rgb image to grayscale
# from PS1 helper code
def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# find area of intersection between a candidate bounding box and an object being tracked (object of interest)
def findIntersection(r, c, width, height, obj, frame_height, frame_width, isBox):
    captionPolygon = Polygon([(c,r), (c, r + height), (c + width, r + height), (c + width, r)])
    objPolygon = None
    if isBox:
        x,y,width,height = obj
        objPolygon = Polygon([(x, y), (x + obj[2], y), (x + obj[2], y + obj[1]), (x, y + obj[1])])
        obj = [[x, y],[x + obj[2], y],[x + obj[2], y + obj[1]],[x, y + obj[1]]]
    else:
        xs = np.array([p[0] for p in obj])
        ys = np.array([p[1] for p in obj])
        minxbound = np.min(xs)
        minybound = np.min(ys)
        maxwidth = np.max(xs) - minxbound
        maxheight = np.max(ys) - minybound
        objPolygon = Polygon([(minxbound, minybound), (minxbound + maxwidth, minybound), (minxbound + maxwidth, minybound - maxheight), (minxbound, minybound - maxheight)])

    pointdists = np.array([captionPolygon.distance(Point(point[0], point[1])) for point in obj])
    minn = np.min(pointdists)
    maxx = np.max(pointdists)
    dist = (minn + maxx)/2 * 10000000
    return captionPolygon.intersection(objPolygon).area + dist


# frame: M x N x 3 source image
# box_size: [width, height]
# obj: list which contains points on boundary of object of interest (basically, don't place captions here) - can extend this later
# if there are no objects being tracked within a frame, pass obj = None (and isBox is False)
# isBox: whether you are feeding in the bounding box (tl-coordinate plus width and height) instead of boundary points (this is true if this is the case)
# and obj is expected to be a list [x,y,width,height] as used in SmartCaptions.py
# return top k top-left corners that correspond to top k found candidate bounding box locations for captions

def rankBoxesFast(frame, boxSize, k=1, energyBlur=32, captionId=None, objBox=None, rescaleFactor=8):
    frame = frame[::rescaleFactor, ::rescaleFactor, ...]
    boxSize = [x // rescaleFactor for x in boxSize]
    if objBox is not None:
        objBox = [x // rescaleFactor for x in objBox]
    return [(x * rescaleFactor, y * rescaleFactor) for x, y in rankBoxes(frame, boxSize, k, energyBlur / rescaleFactor, captionId, objBox)]

def rankBoxes(frame, boxSize, k=1, energyBlur=32, captionId=None, objBox=None, prevPositions={}):
    grayFrame = rgb2gray(frame)
    grayFrame = ndimage.gaussian_filter(grayFrame, 1.5, mode='nearest')
    height, width = boxSize
    halfHeight, halfWidth = math.ceil(height / 2), math.ceil(width / 2)

    frameHeight, frameWidth, _ = frame.shape

    # differentiation filter
    filterX = [[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]]
    filterY = [[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]]
    gx = ndimage.correlate(grayFrame, filterX, mode='constant', cval=0)
    gy = ndimage.correlate(grayFrame, filterY, mode='constant', cval=0)
    
    energy = np.abs(gx) + np.abs(gy)
    
    if captionId is not None and captionId in prevPositions:
        energy[prevPositions[captionId]] -= energyBlur ** 2 * 128
    
    
    if objBox is not None:
        objX, objY, objWidth, objHeight = objBox
        midX, midY = objX + objWidth // 2, objY + objHeight // 2
        midX, midY = max(0, min(frameWidth - 1, midX)), max(0, min(frameHeight - 1, midY))
        energy[objY, midX] -= 512 * energyBlur ** 2
    
    energy = ndimage.gaussian_filter(energy, energyBlur, mode='constant', cval=0)
    
    boxFilter = np.ones(boxSize)
    filtered = ndimage.correlate(energy, boxFilter, mode='constant', cval=0)
    
    if objBox is not None:
        objX, objY, objWidth, objHeight = objBox
        midX, midY = objX + objWidth // 2, objY + objHeight // 2
        midX, midY = max(0, min(frameWidth - 1, midX)), max(0, min(frameHeight - 1, midY))
        #energy[midY, midX] -= 512 * energyBlur ** 2
        filtered[:, :min(midX - objWidth, frameWidth - 1)] += 10000
        filtered[:, max(midX + objWidth, 0):] += 10000
        filtered[:min(objY - objHeight, frameHeight - 1), :] += 10000
        filtered[max(objY + objHeight, 0):, :] += 10000
    
    filtered[:halfHeight, :] = np.inf
    filtered[frameHeight - halfHeight:, :] = np.inf
    filtered[:, :halfWidth] = np.inf
    filtered[:, frameWidth - halfWidth:] = np.inf
    
    #print(np.unravel_index(np.argmin(filtered), filtered.shape))
    
    #plt.imshow(filtered)
    #plt.show()
    
    bestY, bestX = np.unravel_index(np.argmin(filtered), filtered.shape)
    if bestX == 0 and bestY == 0:
        plt.imshow(filtered)
        plt.show()
        print(halfHeight, frameHeight, halfWidth, frameWidth, filtered.shape, filtered[bestY, bestX])
        #print(bestX, bestY, objBox)
    if captionId is not None:
        prevPositions[captionId] = (bestY, bestX)
    
    return [(bestX - halfWidth, bestY - halfHeight)]
    #return [(0, 0)]
    
    '''

    # cumulative sum by rows then columns
    fingrad = np.cumsum(np.cumsum(gradmod, axis = 0), axis = 1)
    maxx = np.amax(fingrad)
    minn = np.amin(fingrad)
    # list of [weighted linear combination of respective innersum and intersection with objects of interest, top left coords of candidate bounding box]
    list_of_positions = []

    actualr = max(0, startr - 3*height)
    actualc = max(int(startc - width/2), 0)
    upperR = startr - height
    
    
    
    
    
    
    list_of_positions.sort()
    # print(list_of_positions)
    return list_of_positions[:k]
    '''


# if you want to test on some frames using select_roi, use the below main method
# def main():
#     # mat = scipy.io.loadmat("twoFrameData", appendmat=True)

#     # im1 = np.array(mat["im2"], dtype = np.uint8)
#     im1 = imageio.imread("cookiemonster3.jpg")
#     fig, ax = plt.subplots(1)

#     ax.imshow(im1)

#     # uncomment the bottom 5 lines to select region, then comment out these lines and run again with new "regiontesting.npy"
#     # roi = roipoly(color="r")
#     # region_points = np.array(list(zip(roi.all_x_points, roi.all_y_points)))
#     # np.save("regiontesting.npy", region_points)
#     k = 1
#     height = 50
#     width = 300
#     region_points = np.load("regiontesting.npy")
#     output = rankBoxes(im1, np.array([width, height]), region_points, k, False)
#     for idx in range(k):
#         r = output[idx][1]
#         rectangle = Rectangle((r[1],r[0]),width,height,linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(rectangle)
    
#     # rectangle = Rectangle((100,50),width,height,linewidth=1,edgecolor='r',facecolor='none')
#     # ax.add_patch(rectangle)
#     # xs = np.array([p[0] for p in region_points])
#     # ys = np.array([p[1] for p in region_points])
#     # cols = int(np.max(xs) - np.min(xs))
#     # rows = int(np.max(ys) - np.min(ys))
#     # startr = int(np.min(ys))
#     # startc = int(np.min(xs))
#     # ax.plot(startr, startc, 'ro')
#     # ax.plot(startr + rows, startc + cols, 'ro')
#     # rectangle = Rectangle((startr,startc),rows,cols,linewidth=1,edgecolor='r',facecolor='none')
#     # ax.add_patch(rectangle)
#     # rectangles go by x,y instead of row, col
#     # rectangle = Rectangle((startc, startr),10,10,linewidth=1,edgecolor='r',facecolor='none')
#     # ax.add_patch(rectangle)
#     plt.show()

# if __name__ == "__main__":
#     main()
