import numpy as np
from scipy import ndimage
from copy import deepcopy
from shapely.geometry import Polygon
import scipy.io
import os
import matplotlib.pyplot as plt
from selectRegion import roipoly
from matplotlib.patches import Rectangle

# convert rgb image to grayscale
# from PS1 helper code
def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# find area of intersection between a candidate bounding box and an object being tracked (object of interest)
def findIntersection(r, c, width, height, obj):
    ys = np.array([p[0] for p in obj])
    xs = np.array([p[1] for p in obj])
    minxbound = np.min(xs)
    minybound = np.min(ys)
    maxwidth = np.max(xs) - minxbound
    maxheight = np.max(ys) - minybound

    # counterclockwise order
    captionPolygon = Polygon([(r,c), (r + height, c), (r + height, c + width), (r, c + width)])
    objPolygon = Polygon([(minxbound, minybound), (minxbound + maxwidth, minybound), (minxbound + maxwidth, minybound + maxheight), (minxbound, minybound + maxheight)])

    return captionPolygon.intersection(objPolygon).area


# frame: M x N x 3 source image
# box_size: [width, height]
# obj: list which contains points on boundary of object of interest (basically, don't place captions here) - can extend this later

# return top k top-left corners that correspond to top k found candidate bounding box locations for captions

def rankBoxes(frame, boxSize, obj, k):
    grayframe = rgb2gray(frame)
    width = boxSize[0]
    height = boxSize[1]
    # uncomment to consider entire frame
    # rows = frame.shape[0]
    # cols = frame.shape[1]

    # uncomment to consider a particular region
    rows = min(frame.shape[0], (np.max(np.array([p[0] for p in obj])) + height))
    cols = min(frame.shape[1], (np.max(np.array([p[1] for p in obj])) + width))
    startr = max(0, (np.min(np.array([p[0] for p in obj])) - height))
    startc = max(0, (np.min(np.array([p[1] for p in obj])) - width))
    # differentiation filter
    filter_x = [[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]]
    filter_y = [[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]]
    gx = ndimage.correlate(grayframe, filter_x, mode = 'constant', cval = 0)
    gy = ndimage.correlate(grayframe, filter_y, mode = 'constant', cval = 0)
    avggrad = (width/(width + height))*gx + (height/(width + height))*gy
    gradmod = deepcopy(avggrad)

    # cumulative sum by rows then columns
    fingrad = np.cumsum(np.cumsum(gradmod, axis = 0), axis = 1)

    # list of [weighted linear combination of respective innersum and intersection with objects of interest, top left coords of candidate bounding box]
    list_of_positions = []

    for r in range(startr, rows):
        for c in range(startc, cols):
            innersum = 0
            if (r + height - 1 >= rows or c + width - 1 >= cols):
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
            deg = findIntersection(r, c, width, height, obj)
            final = alpha*innersum + (1-alpha)*deg
            list_of_positions.append([final, [r, c]])

    list_of_positions.sort()
    return list_of_positions[:k]


# if you want to test on Friends Frames, use the below main method
# def main():
#     mat = scipy.io.loadmat("twoFrameData", appendmat=True)

#     im1 = np.array(mat["im1"], dtype = np.uint8)

#     fig, ax = plt.subplots(1)

#     ax.imshow(im1)

#     # uncomment the bottom 5 lines to select region, then comment out these lines and run again with new "regiontesting.npy"
#     # roi = roipoly(color="r")
#     # indices = roi.get_indices(im1, mat["positions1"])
#     # np.save("pointstesting.npy", indices)
#     # region_points = np.array(list(zip(roi.all_x_points, roi.all_y_points)))
#     # np.save("regiontesting.npy", region_points)
#     k = 5
#     region_points = np.load("regiontesting.npy")
#     output = rankBoxes(im1, np.array([300, 50]), region_points, k)
#     for idx in range(k):
#         r = output[idx][1]
#         rectangle = Rectangle((r[0],r[1]),300,50,linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(rectangle)
#     plt.show()

# if __name__ == "__main__":
#     main()