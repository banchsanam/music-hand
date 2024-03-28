import cv2
import sys
import os
import numpy as np
import musicalbeeps
from PIL import Image, ImageDraw

def getColorMask(img):
    lowerBound = np.array([40, 40, 40])
    upperBound = np.array([70, 255,255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lowerBound, upperBound)

box_arr = []
player = musicalbeeps.Player(volume = 0.3, mute_output = False)
PLAYER_FLAG = True
PLAYER_FLAG2 = True

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
while 1:
    ret, img = cap.read()
    result = cv2.bitwise_and(img, img, mask=getColorMask(img))
    end1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    end3 = cv2.morphologyEx(end1, cv2.MORPH_OPEN, kernel)
    ret, end2 = cv2.threshold(end3, 0, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(end3, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #img is 1280 x 720
    box_height = 160
    box_width = 160

    #drawing all the boxes
    for c in range(0, edged.shape[0], box_width):
        for r in range(0, edged.shape[1], box_height):
            cv2.rectangle(end2, (r, c), (r+box_height, c+box_width), color=(255, 0, 0), thickness=2)
            #going to save each box as a tuple with (x, y, width, height)
            box = (r, c, box_width, box_height)
            box_arr.append(box)

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None:
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0

        cv2.circle(end2, (cX, cY), 5, (255, 0, 0), -1)

        for box in box_arr:
            x, y, w, h = box
            if cX > x and cX < x+w and cY > y and cY < y+h:
                cv2.rectangle(end2, (x, y), (x+w, y+h), color=(200, 50, 50), thickness=5)
                if x==0 and y==0 and PLAYER_FLAG==True:
                    player.play_note("A", 3.5)
                    PLAYER_FLAG = False
                elif x==160 and y==160 and PLAYER_FLAG2==True:
                    player.play_note("C", 3.5)
                    PLAYER_FLAG2 = False

    cv2.imshow('image', end2)
    #print (cv2.countNonZero(end2))
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break