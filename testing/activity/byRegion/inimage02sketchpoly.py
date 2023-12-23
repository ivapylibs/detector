#!/usr/bin/python
#=============================== inimage02sketchpoly ===============================
##
# @addtogroup   Detector_Testing
# @{
# @file   inimage02sketchpoly.py
# @brief  Test basic imageRegions activity region specification from user input.
# 
# User can select polygonal image regions to serve as activity sensitive locations.
# 
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/12/19
#
# @ingroup  Detector_Testing_Activity
# @}
# @quitf
#
# NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#=============================== inimage02sketchpoly ===============================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display



#==[1] Testing/demonstration code to isntantiate and specify activity regions.
#

print("[1] Construct based on specified image.")
theActivity = regact.imageRegions(np.zeros([200,400]))

theImage = np.zeros([200,400,3])

theActivity.specifyPolyRegionsFromImageRGB(theImage)
theActivity.display_cv()

print("Sending signals and testing activity states. Outcome depends on user input.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[25],[25]])
print(theActivity.x)
theActivity.process([[100],[75]])
print(theActivity.x)
theActivity.process([[150],[50]])
print(theActivity.x)

display.wait()

#
#=============================== inimage02sketchpoly ===============================
