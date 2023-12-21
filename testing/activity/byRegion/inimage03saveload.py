#!/usr/bin/python
#=============================== inimage03saveload ===============================
"""!
@brief  Test basic inImage save/load routings. All hard coded, no user input.
"""
#=============================== inimage03saveload ===============================
"""!
@file       inimage03saveload.py

@author     Patricio A. Vela,       pvela@gatech.edu
@date       2023/12/21
"""

#
# NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#=============================== inimage03saveload ===============================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display



#==[2] Specify the regions as a list of polygon arrays.
#
print("[1] Construct based on specified polygons, save, then load and apply.")

thePolygons = ( np.array( [[20, 20, 39, 39],[20, 39, 39, 20]] ),
                np.array( [[60, 60, 79, 79],[60, 79, 79, 60]] ),
                np.array( [[130, 179, 179, 130],[30, 30, 69, 69]] ))


region01 = np.zeros([100,200])  # height x width = flipped.
region01[20:40,20:40]   = 1
region01[60:80,60:80]   = 2
region01[30:70,130:180] = 3

theActivity = regact.inImage.buildFromPolygons([200,400], thePolygons)

theActivity.save('inimage03data.hdf')
theActivity.wipeRegions()

#==[2] Testing/demonstration code to instantiate and specify activity regions.
#
theActivity = regact.inImage.load('inimage03data.hdf')

theActivity.display_cv()

print("Sending signals and testing activity states. Outcomes: 0, 1, 0, 3 .");
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
#=============================== inimage03saveload ===============================
