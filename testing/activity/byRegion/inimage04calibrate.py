#!/usr/bin/python
#================================ inimage04calibrate ===============================
##
# @addtogroup   Detector_Testing
# @{
# @file       inimage04calibrate.py
# @brief  Test out calibration routine that gets user input and saves regions.
# 
# This code tests the case that an initial region mask is available.  In this case,
# it provides one pre-existing region. The calibration routine then adds additional
# regions based on the user input.
# 
# The version without initial regions was manually tested and works. The conditional
# logic is correct.
# 
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/12/21
# @ingroup  Detector_Testing_Activity
# @hidegroupgraph
# @}
# @quitf
#
#! NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#================================ inimage04calibrate ===============================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display



#==[1] Testing/demonstration code to isntantiate and specify activity regions.
#

print("[1] Construct based on specified image, initialized region, and annotations.")
print("    Saving to file.")
theActivity = regact.imageRegions(np.zeros([200,400]))

theImage = np.zeros([200,400,3])
theImage[20:150, 20:150, :] = 200

initRegions = np.zeros([200,400], dtype=int)
initRegions[20:150, 20:150] = 1

regact.imageRegions.calibrateFromPolygonMouseInputOverImageRGB(\
                                        theImage,'inimage04data.hdf', initRegions)


print("[2] Load from file and apply.")
theActivity = regact.imageRegions.load('inimage04data.hdf')

theActivity.display_cv(window_name = "Loaded")


print("    Sending signals and testing activity states. Outcome depends on user input.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[50],[50]])
print(theActivity.x)
theActivity.process([[200],[75]])
print(theActivity.x)
theActivity.process([[300],[75]])
print(theActivity.x)

display.wait()

#
#================================ inimage04calibrate ===============================
