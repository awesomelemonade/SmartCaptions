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

# convert rgb image to grayscale
# from PS1 helper code
def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# find area of intersection between a candidate bounding box and an object being tracked (object of interest)
def findIntersection(r, c, width, height, obj, frame_height, frame_width):
    xs = np.array([p[0] for p in obj])
    ys = np.array([p[1] for p in obj])
    minxbound = np.min(xs)
    minybound = np.min(ys)
    maxwidth = np.max(xs) - minxbound
    maxheight = np.max(ys) - minybound

    captionPolygon = Polygon([(c,r), (c, r + height), (c + width, r + height), (c + width, r)])
    objPolygon = Polygon([(minxbound, minybound), (minxbound + maxwidth, minybound), (minxbound + maxwidth, minybound - maxheight), (minxbound, minybound - maxheight)])

    pointdists = np.array([captionPolygon.distance(Point(point[0], point[1])) for point in obj])
    minn = np.min(pointdists)
    maxx = np.max(pointdists)
    dist = (minn + maxx)/2 * 100000
    return captionPolygon.intersection(objPolygon).area + dist


# frame: M x N x 3 source image
# box_size: [width, height]
# obj: list which contains points on boundary of object of interest (basically, don't place captions here) - can extend this later
# if there are no objects being tracked within a frame, pass obj = None 

# return top k top-left corners that correspond to top k found candidate bounding box locations for captions

def rankBoxes(frame, boxSize, obj, k):
    grayframe = rgb2gray(frame)
    width = boxSize[0]
    height = boxSize[1]


    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # uncomment to consider a particular region
    xs = np.array([p[0] for p in obj])
    ys = np.array([p[1] for p in obj])
    print(np.max(xs))
    print(np.min(xs))
    print(np.max(ys))
    print(np.min(ys)) 


    cols = int(np.max(xs) - np.min(xs))
    rows = int(np.max(ys) - np.min(ys))
    startr = int(np.min(ys))
    startc = int(np.min(xs))
    # differentiation filter
    filter_x = [[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]]
    filter_y = [[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]]
    gx = ndimage.correlate(grayframe, filter_x, mode = 'constant', cval = 0)
    gy = ndimage.correlate(grayframe, filter_y, mode = 'constant', cval = 0)
    avggrad = (height/(width+height))*gy + (width/(width+height))*gx
    gradmod = deepcopy(avggrad)

    # cumulative sum by rows then columns
    fingrad = np.cumsum(np.cumsum(gradmod, axis = 0), axis = 1)
    maxx = np.amax(fingrad)
    minn = np.amin(fingrad)
    # list of [weighted linear combination of respective innersum and intersection with objects of interest, top left coords of candidate bounding box]
    list_of_positions = []
    print(startr)
    print(startr - height)
    print(startc)
    print(startc - width)
    # actualr = max(startr - height, 0)
    # actualc = max(startc - width, 0)
    # upperR = min(startr + rows + height, frame.shape[0])
    # upperC = min(startc + cols + width, frame.shape[1])

    actualr = 0
    actualc = max(startc - width, 0)
    upperR = startr
    if (upperR < height):
        #width-neighborhood, top 1/3 of height of object
        upperR = rows/3
    upperC = min(startc + cols + width, frame.shape[1])

    if obj is None:
        # no objects being tracked, just use entire frame
        rows = frame.shape[0]
        cols = frame.shape[1]
        actualr = 0
        upperR = rows
        actualc = 0
        upperC = cols

    for r in range(actualr, upperR):
        for c in range(actualc, upperC):
            innersum = 0
            if (r + height - 1 >= upperR or c + width - 1 >= upperC):
                # 2d subarray out of bounds, just continue
                continue
            if (r == 0):
                if (c == 0):
                    innersum = fingrad[r + height - 1][c + width - 1]
                else:
                    innersum = fingrad[r + height - 1][c + width - 1] - fingrad[r + height - 1][c -1]
            else:
                if (c == 0):
                    innersum = fingrad[r + height - 1][c + width - 1] - fingrad[r - 1][c + width - 1]
                else:
                    innersum = fingrad[r + height - 1][c + width - 1] - fingrad[r - 1][c + width - 1] - fingrad[r + height - 1][c - 1] + fingrad[r - 1][c - 1]
    
            # weight linear combination of innersum and degree of intersection with objects of interest
            alpha = 0.5
            innersum = (innersum - minn)/(maxx - minn)
            deg = findIntersection(r, c, width, height, obj, frame_height, frame_width)
            final = alpha*innersum + (1-alpha)*deg
            list_of_positions.append([final, [r, c]])

    list_of_positions.sort()
    print(list_of_positions[0][0])
    return list_of_positions[:k]


# if you want to test on some frames using select_roi, use the below main method
# def main():
#     # mat = scipy.io.loadmat("twoFrameData", appendmat=True)

#     # im1 = np.array(mat["im2"], dtype = np.uint8)
#     im1 = imageio.imread("cookiemonster2.jpg")
#     fig, ax = plt.subplots(1)

#     ax.imshow(im1)

#     # uncomment the bottom 5 lines to select region, then comment out these lines and run again with new "regiontesting.npy"
#     # roi = roipoly(color="r")
#     # region_points = np.array(list(zip(roi.all_x_points, roi.all_y_points)))
#     # np.save("regiontesting.npy", region_points)
#     k = 5
#     height = 50
#     width = 500
#     region_points = np.load("regiontesting.npy")
#     output = rankBoxes(im1, np.array([width, height]), region_points, k)
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