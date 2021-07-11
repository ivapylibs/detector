#!/usr/bin/python3
#============================ image04_inRange ============================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image. The depth image
#           is from a depth sensor and saved to a .npz file. We put the
#           preprocesing code in the testing file for now.
#
#
#============================ image04_inRange ============================

# 
# @file     image04_inRange.m
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
#============================ image04_inRange ============================

#==[0] Prep the environment. From most basic to current implementation.
#

import numpy as np
import cv2
import improcessor.basic as improcessor
import detector.inImage as detector
import os

# Get the bound (0.05, 0.95) for the image sequence
def get_hist(depth_frames_raw):
  depth_frame = depth_frames_raw.flatten()
  N = depth_frame.size
  sorted_values = np.sort(depth_frame)
  Tlo = sorted_values[int(N * .05)]
  Thi = sorted_values[int(N * .95)]
  return Tlo, Thi

#==[1] Create a simple image that can be thresholded. Define threshold
#
fpath = os.path.realpath(__file__)
cpath = fpath.rsplit('/', 1)[0]

# Load depth data
depth_frames_raw = np.load(cpath+"/data/depth_raw.npz")["depth_frames"]
N = depth_frames_raw.shape[0]

Tlo, Thi = get_hist(depth_frames_raw)
depth_frames_proc = np.asarray(depth_frames_raw).astype('uint8')
preprocess = improcessor.basic(
                  improcessor.basic.clip, (np.array([Tlo, Thi]),),
                  improcessor.basic.scale, (np.array([0, 255]),),
                  improcessor.basic.to_uint8,(),
                  )

for idx in range(N):
  # preprocess
  depth_frame = preprocess.apply(depth_frames_raw[idx, :, :])
  depth_frames_proc[idx, :, :] = depth_frame

# Pick up one of the frames with a hand hovering on the tabletop
Image = depth_frames_proc[45]

# @todo
# Need the camera pose to transform the image to a top-down view to have a better result

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#

improc = improcessor.basic(cv2.inRange,(30,120,))
binDet = detector.inImage(improc)


#==[3] Apply and visualize.
#
binDet.process(Image)

#==[4] Visualize the output
#
print("Creating window: should see a hand mask (white region).")
cv2.imshow('Output',binDet.Ip.astype(np.uint8))
cv2.waitKey()

#
#============================ image04_inRange ============================
