#!/usr/bin/python
#================================= occupancy01basics =================================
## @file   
# @brief  Test basic imageRegions activity region specification.
# 
# @author     Patricio A. Vela,       pvela@gatech.edu
# @date       2023/12/19
# @ingroup    TestDet_Activity
# @quitf
# NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#================================= occupancy01basics =================================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display
import ivapy.test.vision as imdraw



#==[1] Testing/demonstration code to isntantiate and specify activity regions.
#

print("[1] Construct based on specified image.")
imsize = [100,200]
region01 = np.zeros(np.concatenate((imsize, [3]), axis=0), dtype='bool')  
region01[20:40,20:40,0]   = True
region01[60:80,60:80,1]   = True
region01[30:70,130:180,2] = True

theActivity = regact.imageOccupancy(region01)

theActivity.display_cv()

print("Sending binary images that trigger states in order: None, 1, 2, 3, None.");

testIm = np.full(imsize, False, dtype='bool')
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [True, False, False]
testIm = imdraw.squareInImage(imsize, np.array([25, 25]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, True, False]
testIm = imdraw.squareInImage(imsize, np.array([60, 60]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, False, True]
testIm = imdraw.squareInImage(imsize, np.array([160, 40]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, False, False]
testIm = imdraw.squareInImage(imsize, np.array([180, 80]), 4, [1])
display.binary(testIm)
theActivity.process(testIm)
print(theActivity.z)


display.wait()
display.close("Binary")


print("\n\n[2] Wipe activity regions. Confirm False result.")
theActivity.wipeRegions()

testIm = imdraw.squareInImage(imsize, np.array([25, 25]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

theActivity.display_cv()
display.wait()

print("\n\n[3] Empty activity regions. Confirm False result.")
theActivity.emptyRegions()

testIm = imdraw.squareInImage(imsize, np.array([25, 25]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)


print("\n\n[3] Adding polygonal regions (equal to first case).")
theActivity.addRegionByPolygon(np.array([[20,20,39,39],[20,39,39,20]]), imsize)
theActivity.addRegionByPolygon(np.array([[60,60, 79, 79],[60,79,79,60]]))
theActivity.addRegionByPolygon(np.array([[130,179,179,130],[30,30, 69, 69]]))

theActivity.display_cv()

print("Sending signals that trigger states in order: None, 1, 2, 3, None");
testIm = np.full(imsize, False, dtype='bool')
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [True, False, False]
testIm = imdraw.squareInImage(imsize, np.array([25, 25]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, True, False]
testIm = imdraw.squareInImage(imsize, np.array([60, 60]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, False, True]
testIm = imdraw.squareInImage(imsize, np.array([160, 40]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

# Outcome: [False, False, False]
testIm = imdraw.squareInImage(imsize, np.array([180, 80]), 4, [1])
theActivity.process(testIm)
print(theActivity.z)

display.wait()
quit()


display.wait()


#
#================================= inimage01basics =================================
