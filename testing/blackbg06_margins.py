#================================ blackbg06_margins ================================
'''!
@brief  Use Realsense data to collect background classifier margin values across
        the image and over time.  Stores the maximum value read while collecting
        data.

  Works for D435i or equivalent RGB-D camera from Intel Realsense line.  Builds on 
  blackbg04 script.  This version applies hard coded values for the detector, but
  monitors the classification margin over the image and keeps track of the maximum 
  value.  For those pixels that are part of the workspace, the maximum observed
  margin should reflect the threshold shift needed per pixel to recover a spatially
  variable threshold.

  In practice, it should be possible to have tau be image shaped so that the
  classifier can indeed apply a spatially variable threshold.  That should be
  done elsewhere (another test script or directly in the code).  

  A further test script should also play around with saving and loading this
  information.  Makes sense to create this script first and then use it to
  load and apply the spatially variable threshold.  Once done, can be included in 
  the actual codebase for this package.
'''
#================================ blackbg06_margins ================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
## Code modified from librealsense library.
## https://github.com/IntelRealSense/librealsense
##
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#
#================================ blackbg06_margins ================================


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
print('Starting ... Use Ctrl-C to Quit.')

mI = None

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

        # Perform margin computation and stored value updates.
        if mI is None:
          mI = bgDetector.calc_margin(color_image)
        else:
          cmI = bgDetector.calc_margin(color_image)
          mI  = np.maximum(mI, cmI)

        val = cv2.waitKey(1)
        if val == 113:
            print("Closing shop")
            break


finally:
    # Stop streaming and display result (needs shift + rescaling)
    pipeline.stop()

    minV = mI.min()
    maxV = mI.max()
    print("Minimum value: "), print(minV)
    print("Maximum value: "), print(maxV)

    if (maxV == minV):
      maxV = minV + 1
    scI = np.subtract(mI, minV)/(maxV-minV)
    cv2.namedWindow('Max Margin Observed', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Max Margin Observed', scI.astype(float))
    cv2.waitKey(-1)

#
#================================ blackbg06_margins ================================
