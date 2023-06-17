#!/usr/bin/python
#============================== tgm03depth435 ==============================
"""
@brief          Real-time implementation of red glove segmentation.

Expands on the tgm02 script by applying to image sequence.  There are two
phases of operation.  The first does a little model updating until the
`q` key is pressed.  The second freezes the glove target model and performs
color-based detection.  Hitting `q` again quits the routine.
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
import detector.fgmodel.Gaussian as SGM 


#--[0]  Setup the camera and the red glove target model.
#       Use hard coded glove configuration.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('tgm03depth435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()

fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([1100.0,250.0,250.0]) )
fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

#--[1] Run the red glove detector with model updating until `q` pressed.
#
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

#--[2] Run red glove detector with frozen model until `q` pressed.
#
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
#============================== tgm03depth435 ==============================
