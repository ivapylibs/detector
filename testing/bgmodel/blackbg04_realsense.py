#!/usr/bin/python
#=============================== blackbg04_realsense ===============================
'''!
@brief  Simple code with original Realsense API to snag color imagery and apply the
        planar black background background estimation model. Visualize in window.

Works for D435i or equivalent RGB-D camera from Intel Realsense line.  The
code demonstrates a couple of things.  First how to use the realsense in python to 
snag imagery.  Second, how to setup the realsense configuration by loading a
pre-existing JSON file. Lastly, it applies the black background detector model
to segment out parts of the scene that are considered to be "background."  In
this case, the premise is that the camera looks down at a black mat that defines
the operational workspace associated to the task.  Anything in the mat area that is
not black is an object of interest.

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the original image plus the masked image (full size).
The black background should be solid black.
Hit "q" to quit.

'''
#=============================== blackbg04_realsense ===============================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/20

'''
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
## Code modified from librealsense library.
## https://github.com/IntelRealSense/librealsense
##
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#
#=============================== blackbg04_realsense ===============================


import pyrealsense2 as rs
import numpy as np
import cv2
import json

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

# Start streaming
pipeline.start(config)
print('Starting ... Press "q" to Quit.')


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

        bgM = cv2.cvtColor(255-bgmask.x.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        fgI = cv2.bitwise_and(color_image, bgM)

        # Show images
        cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Image', color_image)

        cv2.namedWindow('Masked Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Masked Image', fgI)
        opKey = cv2.waitKey(1)
        if opKey == ord('q'):
            break


finally:
    # Stop streaming
    pipeline.stop()

#
#=============================== blackbg04_realsense ===============================
