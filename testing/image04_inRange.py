#!/usr/bin/python
#============================ image04_inRange ============================
## @file
# @brief    inImage detector using OpenCV inRange with stored depth data.
#
# Test out the in image detector for a depth image saved from a depth sensor
# and saved to a .npz file. The preprocesing code is the testing file.
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @author   Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2021/07/03 [created]
# @date     2021/07/11 [modified]
# @ingroup  TestDetector
# @quitf
#
# NOTE:
#   Indent is set to 2 spaces.
#   Tab is set to 4 spaces with conversion to spaces.
#
#============================ image04_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#

import numpy as np
import cv2
import improcessor.basic as improcessor
import detector.inImage as detector
import os


#==[1] Create a simple image that can be thresholded. Define threshold
#
fpath = os.path.realpath(__file__)
cpath = fpath.rsplit('/', 1)[0]

# Load depth data
# npz file may be corrupted by git lfs , See https://stackoverflow.com/a/65414582/5269146
depth_frames_raw = np.load(cpath+"/data/depth_raw.npz",allow_pickle=True,fix_imports=True,encoding='latin1')["depth_frames"]
N = depth_frames_raw.shape[0]

# Pick up one of the frames with a hand hovering on the tabletop
Image = depth_frames_raw[60]

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#

improc = improcessor.basic(cv2.inRange,(0.4,0.7,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#
binDet.process(Image)

#==[4] Visualize the output
#
print("Creating window: should see a noisy hand mask (white region).")
cv2.imshow('Output',binDet.getState().x.astype(np.uint8))
cv2.waitKey()

#
#============================ image04_inRange ============================
