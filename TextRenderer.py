# https://www.geeksforgeeks.org/python-opencv-write-text-on-video/

import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT = font = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FOREGROUND_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
DEBUG_COLOR = (0, 255, 0) # GREEN
debug = False

# renders text centered around the box
def renderCaption(frame, box, message, scale=FONT_SCALE):
    x, y, width, height = box
    centerX, centerY = x + width // 2, y + height // 2
    cv2.rectangle(frame, (x, y), (x + width, y + height), BACKGROUND_COLOR, cv2.FILLED)
    
    (textWidth, textHeight), baseline = cv2.getTextSize(message, FONT, scale, FONT_THICKNESS)
    textOriginX, textOriginY = centerX - textWidth // 2, centerY - (textHeight + baseline) // 2
    
    cv2.putText(frame, text=message, org=(textOriginX, textOriginY + textHeight), fontFace=FONT,
            fontScale=scale, color=FOREGROUND_COLOR, thickness=FONT_THICKNESS)
    if debug:
        cv2.rectangle(frame, (textOriginX, textOriginY),
                (textOriginX + textWidth, textOriginY + textHeight + baseline), DEBUG_COLOR)

def getCaptionSize(message, scale=FONT_SCALE):
    (textWidth, textHeight), baseline = cv2.getTextSize(message, FONT, scale, FONT_THICKNESS)
    return textWidth, textHeight + baseline
