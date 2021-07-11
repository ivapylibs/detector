#!/usr/bin/python3
#============================ image04_inRange ============================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image.
#
# Upgrades earlier test scripts to use a depth image plus upper and lower
# depth limits to establish a detection depth zone. This depth image is an
# actual one obtained from a depth sensor and saved to a file.
#
#============================ image04_inRange ============================

# 
# @file     image04_inRange.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/03 [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#============================ image04_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#
import operator
import numpy as np

import cv2

import improcessor.basic as improcessor
import detector.inImage as detector


# YUNZHI, MODIFY TO APPLY TO A LOADED RAW DEPTH IMAGE IN THE data
# DIRECTORY.  AIM TO LOAD THE npz VERSION AND NOT THE png VERSION SINCE
# THAT ONE INVOLVED SOME CLIPPPING AND SCALING. BETTER TO USE THE ORIGINAL
# DATA SINCE THAT IS WHAT WE WILL BE USING.
#

#==[1] Create a simple image that can be thresholded. Define threshold
#
image = np.zeros((100,100))
image[1:4,4:7]  = 10
image[4:9,7:20] = 10
# YUNZHI: REPLACE WITH LOAD DATA.

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#
# YUNZHI: IDENTIFY BETTER THRESHOLDS
improc = improcessor.basic(cv2.inRange,(7,15,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#
binDet.process(image)

#==[4] Visualize the output
#
print("Creating window: should see YUNZHI: WHAT WHAT WHAT? KEEP SHORT.")
cv2.imshow('Output',binDet.Ip.astype(np.uint8))
cv2.waitKey()

#
#============================ image04_inRange ============================
