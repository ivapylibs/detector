#!/usr/bin/python
#================================= inimage01basics =================================
"""!
@brief  Test basic inImage activity region specification.

"""
#================================= inimage01basics =================================
"""!
@file       inimage01basics.py

@author     Patricio A. Vela,       pvela@gatech.edu
@date       2023/12/19
"""

#
# NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#================================= inimage01basics =================================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display



#==[1] Testing/demonstration code to isntantiate and specify activity regions.
#

print("[1] Construct based on specified image.")
region01 = np.zeros([100,200])  # height x width = flipped.
region01[20:40,20:40]   = 1
region01[60:80,60:80]   = 2
region01[30:70,130:180] = 3

theActivity = regact.inImage(region01)

theActivity.display_cv()

print("Sending signals that trigger states in order: 0, 1, 2, 3.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[25],[25]])
print(theActivity.x)
theActivity.process([[75],[75]])
print(theActivity.x)
theActivity.process([[150],[50]])
print(theActivity.x)

display.wait()

print("\n\n[2] Wipe activity regions. Confirm none result.")
theActivity.wipeRegions()


theActivity.display_cv()

print("Sending signals that trigger states in order: 0, 0, 0, 0.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[25],[25]])
print(theActivity.x)
theActivity.process([[75],[75]])
print(theActivity.x)
theActivity.process([[50],[150]])
print(theActivity.x)

display.wait()


print("\n\n[3] Adding polygonal regions (equal to first case).")
theActivity.addRegionByPolygon(np.array([[20,20,39,39],[20,39,39,20]]))
theActivity.addRegionByPolygon(np.array([[60,60, 79, 79],[60,79,79,60]]))
theActivity.addRegionByPolygon(np.array([[130,179,179,130],[30,30, 69, 69]]))

theActivity.display_cv()

print("Sending signals that trigger states in order: 0, 1, 2, 3.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[25],[25]])
print(theActivity.x)
theActivity.process([[75],[75]])
print(theActivity.x)
theActivity.process([[150],[50]])
print(theActivity.x)

display.wait()


#
#================================= inimage01basics =================================
