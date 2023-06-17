#!/usr/bin/python
#============================== tgm03depth435 ==============================
"""
@brief          More advanced setup of D435 camera from stored JSON settings.

Expands on the test01 implementation by permitting the loading of setting
information from a YAML file with an option JSON file for more detailed D435
settings (as obtained from the realsense_viewer application).
"""
#============================== tgm03depth435 ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#============================== tgm03depth435 ==============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.fgmodel.Gaussian as SGT 
import detector.bgmodel.Gaussian as SGB 

#--[0]  Setup the camera, red glove target model, and workspace depth model.
#       Use hard coded glove configuration.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('tgm04depth435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()

fgModP  = SGT.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([1100.0, 250.0, 250.0]) )
fgModel = SGT.Gaussian( SGT.CfgSGT.builtForRedGlove(), None, fgModP )

bgModel = SGB.Gaussian( SGB.CfgSGM.builtForDepth435() )

#--[1] Run the depth model estimation and the red glove detector with model
#      updating, sequentially, waiting `q` press to continue.
#
print('Step 1.A is to calibrate the depth model.')

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
                   window_name="Output. Press \'q\' to move on.")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

print('Step 1.B is to calibrate the red glove model.')
print('Hit any key to continue once scene is prepped.')
opKey = cv2.waitKey(0)

while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.process(rgb)
    fgS = fgModel.getState()

    fgIm = cv2.cvtColor(fgS.fgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    display.display_rgb_cv(fgIm, ratio=0.5, \
                   window_name="Output. Press \'q\' to move on.")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
print("Switching adaptation off and running combined method.")


while(True):
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    bgModel.process(dep)
    bgS = bgModel.getState()

    fgModel.detect(rgb, np.logical_not(bgS.bgIm))
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
#============================== tgm03depth435 ==============================
