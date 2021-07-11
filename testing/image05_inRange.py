#!/usr/bin/python3
#============================ image05_inRange ============================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image. The depth
#           image sequence is from a depth sensor and preprocessed to
#           a video file.
#
#============================ image05_inRange ============================

# 
# @file     image05_inRange.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
#           Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2021/07/03 [created]
#           2021/07/11 [modified]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#============================ image05_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#
import operator
import numpy as np

import cv2

import improcessor.basic as improcessor
import detector.inImage as detector
import os

#==[1] Create a video sequence that can be thresholded.
#
fpath = os.path.realpath(__file__)
cpath = fpath.rsplit('/', 1)[0]

cap = cv2.VideoCapture(cpath+'/data/depth_proc.avi')

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#

improc = improcessor.basic(cv2.cvtColor, (cv2.COLOR_BGR2GRAY,),\
                           cv2.inRange,(30, 255,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#

print("Creating window: should see a hand mask (white region) across the image sequence.")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    binDet.measure(frame)
    # Display the resulting frame

    cv2.imshow('Demo', binDet.Ip)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#
#============================ image05_inRange ============================
