#!/usr/bin/python
#============================== pws04calibrate =============================
"""
@brief          Example of onWorkspace calibration then deployment.

More compact calibration, build, and save for loading.  This implementation
is closer to how would be used.


Execution:
----------
Assumes availability of Intel Realsense D435 camera or equivalent.

Operates in two phases.  Calibration, then deployment.  Press 'q' to go from
first phase to second, then to quit.
"""
#============================== pws04calibrate =============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/07/20              [created]
#
#
# NOTE: indent is 4 spaces with conversion. 85 columns.
#
#============================== pws04calibrate =============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.bgmodel.onWorkspace as GWS 

d435_configs = d435.CfgD435()
d435_configs.merge_from_file('sgm02depth435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()

theConfig = GWS.CfgOnWS.builtForDepth435()

bgModel = GWS.onWorkspace.buildAndCalibrateFromConfig(theConfig, theStream, True)
bgModel.save('pws05saved.hdf5')
bgModel = None

print("Saved model. Cleared model. Now loading ...")

bgModel = GWS.onWorkspace.load('pws05saved.hdf5')
bgModel.state = GWS.RunState.DETECT
print("Switching adaptation off for deployment.")

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.process(dep)
    bgS = bgModel.getState()
    bgD = bgModel.getDebug()

    bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    display.rgb_depth_cv(bgIm, dep, ratio=0.5, window_name="RGB+Depth")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

#
#============================== pws04calibrate =============================
