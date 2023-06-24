#!/usr/bin/python
#================================ blackbg07_estimate ===============================
'''!
@brief  Collect from D435 background classifier margin values then apply as threshold.

  Works for D435i or equivalent RGB-D camera from Intel Realsense line.  Builds on 
  blackbg06 script.  This version applies hard coded values for the detector, but
  the uses the inCornerEstimator to monitors the classification margins over time.
  These values are then applied to an inCorner instance for the image stream.

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the original image plus the masked image.
During first phase, collects data for the thresholds. Hit "q" to continue.
During second phase, applies estimated thresholds. Hit "q" to stop.

'''
#================================ blackbg07_estimate ===============================
#
# @author Patricio A. Vela,   pvela@gatech.edu
# @date   2023/04/21
#
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
#================================ blackbg07_estimate ===============================


import numpy as np
import cv2

import camera.utils.display as display
import camera.d435.runner2 as d435
import detector.bgmodel.inCorner as bgdet


d435_configs = d435.CfgD435()
d435_configs.merge_from_file('blackbg07.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()


bgModel = bgdet.inCorner.build_model_blackBG(-70, 0)
bgDetector = bgdet.inCornerEstimator()
bgDetector.set_model(bgModel)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgDetector.process(rgb)
    bgmask = bgDetector.getState()

    display.rgb_binary_cv(rgb, bgmask.x, ratio=0.5, window_name="RGB+Mask")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Grab model then apply.")
        break

bgDetector.apply_estimated_margins()
bgModel = bgDetector.bgModel
bgModel.offsetThreshold(35)

bgDetector = None
bgDetector = bgdet.inCorner(None, bgModel)

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgDetector.process(rgb)
    bgmask = bgDetector.getState()

    display.rgb_binary_cv(rgb, bgmask.x, ratio=0.5, window_name="RGB+Mask")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

#
#================================ blackbg07_estimate ===============================
