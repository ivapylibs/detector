#!/usr/bin/python
#============================ image03_inRange ============================
## @file
# @brief    In image detector with improcessor using OpenCV inRange on fake image.
#
# Upgrades earlier test scripts to use a depth image plus upper and lower
# depth limits to establish a detection depth zone. The "depth image" here
# is simply ficticious data placed into an array (same as fake data in
# earlier test scripts).
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/11 [created]
# @ingroup  TestDetector
# @quitf
#
# NOTE:
#   Indent is set to 2 spaces.
#   Tab is set to 4 spaces with conversion to spaces.
#
#============================ image03_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#

import numpy as np
import cv2
import improcessor.basic as improcessor
import detector.inImage as detector

#==[1] Create a simple image that can be thresholded. Define threshold
#
image = np.zeros((100,100))
image[1:4,4:7]  = 10
image[4:9,7:20] = 10


#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#
improc = improcessor.basic(cv2.inRange,(7,15,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#
binDet.process(image)

#==[4] Visualize the output
#
print("Creating window: should see some true values (two adjacent boxes).")
cv2.imshow('Output',binDet.getState().x.astype(np.uint8))
cv2.waitKey()

#
#============================ image03_inRange ============================
