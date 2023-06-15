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
import detector.fgmodel.Gaussian as SGM 

d435_configs = d435.CfgD435()
d435_configs.merge_from_file('tgm03depth435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()


fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([900.0,250.0,250.0]) )
fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRed(), None, fgModP )


while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.process(rgb)
    fgS = fgModel.getState()

    fgIm = cv2.cvtColor(fgS.fgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    display.display_rgb_cv(fgIm, ratio=0.5, \
                   window_name="Camera signals. Press \'q\' to move on.")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)

print("Switching adaptation off.")

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.detect(rgb)
    fgS = fgModel.getState()

    fgIm = cv2.cvtColor(fgS.fgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    display.display_rgb_cv(fgIm, ratio=0.5, \
                   window_name="Camera signals. Press \'q\' to exit")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
#
#=============================== test02_align ==============================
