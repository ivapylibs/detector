#!/usr/bin/python
#==================================== sgm03color ===================================
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
#==================================== sgm03color ===================================


#==[0]  Setup the environment.
#
import numpy as np
import cv2

import ivapy.display_cv as display
import camera.d435.runner2 as d435
import detector.bgmodel.Gaussian as detect

#==[1]  Instantiation components.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('sgm03color.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()

theCfg  = detect.CfgSGM().builtForBlackMat()

###==[2] Calibrate.
#
bgModel = detect.bgGaussian.buildAndCalibrateFromConfigRGBD(theCfg, None, theStream, True)

#==[3] Apply to scene.
#
bgD = bgModel.getDebug()
bgMod = bgD.mu.astype(np.uint8)
display.rgb(bgMod, ratio=0.5, window_name="BG Model")
print("Displayed the expected image.")
print("Running as detector on scene w/display in new window.")

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.detect(rgb)
    bgS = bgModel.getState()

    display.binary(bgS.bgIm, ratio=0.5, window_name="Detection")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

display.close("Detection")
display.close("BG Model")

#
#==================================== sgm03color ===================================
