#!/usr/bin/python
#=============================== region01_realsense ================================
## @file
# @brief  Background model processed for largest connected region evaluated over time
#         to extract most consistently classified background.  Good for establishing
#         workspace ROI. Anything outside is not of interest.
# 
# Works for D435i or equivalent RGB-D camera from Intel Realsense line.  Builds on
# blackbg05 by adding largest connected component processing using skimage.  While
# OpenCV has similar functions, the skimage ones are much easier to use, have better
# documentation, and better map to Matlab's routines.
# 
# 
# Execution:
# ----------
# Assumes an Intel Realsense D435 or compatible RGBD camera.
# 
# Just run it and hit 'q' when ready to quit model updating.
# Displays largest region in image.  
# Hit 'q' again to quit.
# 
# @author Patricio A. Vela,   pvela@gatech.edu
# @date   2023/04/25
# @ingroup  TestDetector
# @quitf
#
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
# Code modified from librealsense library.
# https://github.com/IntelRealSense/librealsense
#
# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#
#=============================== region01_realsense ================================


import pyrealsense2 as rs
import numpy as np
import cv2
import json
from skimage import measure 
from skimage import morphology 

import detector.bgmodel.inCorner as bgdet

loadConfig = True

# Configure depth and color streams
pipeline = rs.pipeline()
config   = rs.config()

# Get device for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)


if loadConfig:
    adv_mode = rs.rs400_advanced_mode(device)
    if not adv_mode.is_enabled():
        print('Issues with advanced mode')
        quit()
  
    configFile = open('data/setup02.json')
    configStr = json.load(configFile)
    configStr = str(configStr).replace("'", '\"')
    configFile.close()
  
    print('Loaded JSON configuration and applying. Camera + BG Model setup ...')
    adv_mode.load_json(configStr)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
else:
    print('Using current camera configuration')

bgModel = bgdet.inCorner.build_model_blackBG(-70, 0)
bgDetector = bgdet.inCorner()
bgDetector.set_model(bgModel)
roiUnion = None

# Start streaming
pipeline.start(config)
print('Starting ... Hit q to Quit.')


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy array and apply BG detection.
        # Use BG detection to mask out the BG. Only show "foreground."
        color_image = np.asanyarray(color_frame.get_data())

        bgDetector.process(color_image)
        bgmask = bgDetector.getState()

        bglabeled = measure.label(bgmask.x, connectivity=2)
        bgprops   = measure.regionprops_table(bglabeled, properties=['label','area'])

        bigl  = np.argmax(bgprops['area'])
        bgroi = (bglabeled == bgprops['label'][bigl])

        if roiUnion is None:
            roiUnion = bgroi
        else:
            roiUnion = np.logical_and(roiUnion, bgroi)

        bgM = cv2.cvtColor(255-roiUnion.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        fgI = cv2.bitwise_and(color_image, bgM)

        # Show images
        cv2.namedWindow('Masked Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Masked Image', fgI)

        cv2.namedWindow('ROI Mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ROI Mask', bgM)

        val = cv2.waitKey(1)

        if val >= 0:
            if (val == 81) or (val == 44):    # left arrow or left angle bracket.
                bgModel.adjustThreshold( bgModel.tau - 1 )
                bgDetector.set_model(bgModel)
            elif (val == 83) or (val == 46):  # right arrow or right angle bracket.
                bgModel.adjustThreshold( bgModel.tau + 1 )
                bgDetector.set_model(bgModel)
            elif (val == 82) or (val == 91):    # left arrow or left angle bracket.
                bgModel.adjustThreshold( bgModel.tau - 10 )
                bgDetector.set_model(bgModel)
            elif (val == 84) or (val == 93):  # right arrow or right angle bracket.
                bgModel.adjustThreshold( bgModel.tau + 10 )
                bgDetector.set_model(bgModel)
            elif (val == 113):
                break


finally:
    # Stop streaming
    pipeline.stop()

#
#=============================== region01_realsense ================================
