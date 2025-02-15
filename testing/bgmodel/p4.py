#!/usr/bin/python
#============================== pws03saveload ==============================
"""
@brief          Test out model saving with Workspace detection method.

Extends ''pws02depth435'' to include saving of model then loading and
application of the saved model.  


Execution:
----------
Assumes availability of Intel Realsense D435 camera or equivalent.

Operates in two phases.  First phase is "model estimation"/learning.
After that the model is saved, then loaded and applied for detection.  
Press 'q' to go from first phase to second, then to quit.
"""
#============================== pws03saveload ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/06/22              [created]
#
#
# NOTE: indent is 4 spaces with conversion. 85 columns.
#
#============================== pws03saveload ==============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.bgmodel.onWorkspace as GWS 


bgModel = GWS.onWorkspace.load('pws03saved.hdf5')

quit()

bgModel.state = GWS.RunState.DETECT
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

    print("Max error and threshold are: ", np.amax(bgModel.maxE), 
                                           bgModel.config.tauSigma, 
                                           np.amax(bgModel.nrmE))
    print("Max/min depth are:", np.amax(bgModel.measI), np.amin(bgModel.measI))
    display.display_rgb_dep_cv(bgIm, dep, ratio=0.5, \
                   window_name="RGB+Depth signals. Press \'q\' to exit")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

#
#============================== pws03saveload ==============================
