#!/usr/bin/python
#============================ image06_inRange ============================
'''!
@brief    

The depth image sequence is from depth sensor data saved to a .npz file.
The npz file is loaded and processed.

'''
#============================ image06_inRange ============================

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
#============================ image06_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#
import operator
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

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#

improc = improcessor.basic(cv2.inRange,(0.4,0.7,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#
print("Creating window: should see a noisy hand mask (white region).")
for idx in range(N):
  binDet.process(depth_frames_raw[idx, :, :])
  cv2.imshow('Output',binDet.getState().x.astype(np.uint8))

  # Press Q on keyboard to exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break

#
#============================ image06_inRange ============================
