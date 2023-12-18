#!/usr/bin/python
#================================= blackbg10_region ================================
#================================ blackbg06_margins ================================
'''!
@brief  Use Realsense API to collect background classifier margin values across
        imagery over time.  Stores the maximum value read while collecting data.

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


Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the original image plus the masked image (full size).
During running it stores distance/error values observed and keeps track of max value per pixel.
Hit "q" to quit.

'''
#================================= blackbg10_region ================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/07/20

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


import numpy as np
import cv2

import camera.utils.display as display
import camera.d435.runner2 as d435
import detector.bgmodel.inCorner as bgdet


d435_configs = d435.CfgD435()
d435_configs.merge_from_file('blackbg07.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()


bgModel = bgdet.inCorner.build_model_blackBG(-70, 0)

bgDetector = bgdet.inCornerEstimator()
bgDetector.set_model(bgModel)

print("Running detector for a bit. Hit 'q' to move on.")
bgDetector.refineFromRGBDStream(theStream, True)


print("Estimating ROI mask. Hit 'q' to move on.")
theMask = bgDetector.maskRegionFromRGBDStream(theStream, True)


print("Testing out ROI mask. Hit 'q' to move on.")
while(True):
  rgb, dep, success = theStream.get_frames()
  if not success:
    print("Cannot get the camera signals. Exiting...")
    exit()

  bgDetector.process(rgb)
  bgmask = bgDetector.getState()

  display.rgb_binary_cv(rgb, theMask & bgmask.x, ratio=0.5, window_name="RGB+Mask")

  opKey = cv2.waitKey(1)
  if opKey == ord('q'):
    break

#
#================================= blackbg10_region ================================
