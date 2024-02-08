#!/usr/bin/python
#==================================== gmmcv01run ===================================
## @file
# @brief    Use Realsense camera to collect background model data, then apply for
#           foreground detection using OpenCV GMM implementation.
#
#  Works for D435i or equivalent RGB-D camera from Intel Realsense line.
#
#  Execution:
#  ----------
#  Needs Intel Realsense D435 or equivalent RGBD camera.
#
#  Just run and prints out calibrate process information.
#  Once calibrated, runs another loop to apply to live image stream.
#  Hit "q" to quit.
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2024/02/08          [created]
# @ingroup  TestDetector
# @quitf
#
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
#==================================== gmmcv01run ===================================


#==[0]  Setup the environment.
#
import numpy as np
import cv2

import camera.utils.display as display
import camera.d435.runner2 as d435
import detector.bgmodel.inCorner as bgdet


#==[1]  Instantiation components.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('sgm02depth435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()

theCfg  = detct.CfgGMM_cv.builtForLearning()
bgModel = detect.bgmodelGMM_cv(theCfg)


#==[2] Calibrate.
#
bgModel = GWS.onWorkspace.buildAndCalibrateFromConfigRGBD(theConfig, theStream, True)

#==[3] Apply to scene.
#
print("Running as detector only on scene.")

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.detect(rgb)
    bgS = bgModel.getState()
    bgD = bgModel.getDebug()

    bgIm  = cv2.cvtColor(bgS.bgIm, cv2.COLOR_GRAY2BGR)
    bgMod = bgD.mu.astype(np.uint8)

    display.rgb(bgIm, bgD.mu, ratio=0.25, window_name="Detection")
    display.rgb(bgIm, bgD.mu, ratio=0.25, window_name="BG Model")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

display.close_cv("Detection")
display.close_cv("BG Model")

#
#==================================== gmmcv01run ===================================
