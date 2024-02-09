#!/usr/bin/python
#================================= gmmcv02calibrate ================================
## @file
# @brief    Use Realsense camera to collect background model data, save to file, then
#           load file and apply foreground detection using OpenCV GMM implementation.
#
#  Works for D435i or equivalent RGB-D camera from Intel Realsense line.
#
#  Execution:
#  ----------
#  Needs Intel Realsense D435 or equivalent RGBD camera.
#
#  Just run and follow instructions to calibrate.  It will save then load calibration.
#  Once loaded with new instance, runs another loop to apply to live image stream.
#  Hit "q" to quit.
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2024/02/08          [created]
# @ingroup  TestDetector
# @quitf
#
# 
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
#================================= gmmcv02calibrate ================================


#==[0]  Setup the environment.
#
import numpy as np
import cv2

import ivapy.display_cv as display
import camera.d435.runner2 as d435
import detector.bgmodel.GMM as detect


#==[1]  Instantiation components.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('sgm02depth435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()

theCfg  = detect.CfgGMM_cv.builtForLearning()
theCfg.detectShadow = True


#==[2] Calibrate then save.
#
bgModel = detect.bgmodelGMM_cv.buildAndCalibrateFromConfigRGBD(theCfg, theStream, True)
print("Saving model.")
bgModel.save("gmmcv02model.hdf5")
print("Done model.")


#==[3] Load 
#
bgModel = None
bgModel = detect.bgmodelGMM_cv.load("gmmcv02model.hdf5")


#==[4] Apply to live scene.
#
bgD   = bgModel.getDebug()
bgMod = bgD.mu.astype(np.uint8)
display.rgb(bgMod, ratio=0.25, window_name="BG Model")
print("Displayed the expected image.")
print("Running as detector on scene w/display in new window.")

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.detect(rgb)
    bgS = bgModel.getState()

    bgIm  = cv2.cvtColor(bgS.bgIm, cv2.COLOR_GRAY2BGR)

    display.rgb(bgIm, ratio=0.25, window_name="Detection")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

display.close("Detection")
display.close("BG Model")

#
#================================= gmmcv02calibrate ================================
