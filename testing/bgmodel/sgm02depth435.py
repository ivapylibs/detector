#!/usr/bin/python
#=============================== test02_align ==============================
"""
@brief          More advanced setup of D435 camera from stored JSON settings.

Expands on the test01 implementation by permitting the loading of setting
information from a YAML file with an option JSON file for more detailed D435
settings (as obtained from the realsense_viewer application).
"""
#=============================== test02_align ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#=============================== test02_align ==============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.bgmodel.Gaussian as SGM 

d435_configs = d435.CfgD435()
d435_configs.merge_from_file('sgm02depth435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()

bgModel = SGM.Gaussian( SGM.CfgSGM.builtForDepth435() )

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.process(dep)
    bgS = bgModel.getState()
    bgD = bgModel.getDebug()

    bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    display.display_rgb_dep_cv(bgIm, bgD.mu, ratio=0.5, \
                   window_name="Camera signals. (color-scaled depth). Press \'q\' to exit")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

bgModel.config.alpha = 0.0
print("Switching adaptation off.")

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.process(dep)
    bgS = bgModel.getState()
    bgD = bgModel.getDebug()

    bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    print("Counting foreground pixels.", np.count_nonzero(bgS.bgIm))

    print("Max error and threshold are: ", np.amax(bgModel.maxE), bgModel.config.tauSigma, np.amax(bgModel.nrmE))
    print("Max/min depth are:", np.amax(bgModel.measI), np.amin(bgModel.measI))
    display.display_rgb_dep_cv(bgIm, dep, ratio=0.5, \
                   window_name="Camera signals. (color-scaled depth). Press \'q\' to exit")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

#
#=============================== test02_align ==============================
