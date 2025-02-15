#!/usr/bin/python3
#============================== diff01simple =============================
##@file
# @brief    Test script for the most basic functionality of template
#           puzzle piece class.
#
# Places simple shapes into an image and check for image differences after
# each one.  Accumulation is active, which should generate a label image
# for the shapes. 
#
# @ingroup  TestDet_FG
#
# @author   Patricio A. Vela,       pvela@gatech.edu
#
# @date     2025/02/14  
#
# @quitf
#============================== diff01simple =============================

#==[0] Prep environment
import matplotlib.pyplot as plt
import numpy as np

from puzzle.piece import Template
import detector.fgmodel.differences as diffs

# Create empty image, instantiate differences FG detector.
bigImage = np.zeros((200, 200, 3))

cfgDet = diffs.CfgDifferences()
cfgDet.doAccum= True

chDet = diffs.fgDifferences(cfgDet)

chDet.process(bigImage)

# Add shapes to image and generate difference.
squarePiece = Template.buildSquare(20, color=(255, 0, 0), rLoc=(80, 40))
squarePiece.placeInImage(bigImage)

chDet.process(bigImage)

spherePiece = Template.buildSphere(10, color=(0, 255, 0), rLoc=(80, 140))
spherePiece.placeInImage(bigImage)

chDet.process(bigImage)

plt.figure()
plt.imshow(bigImage)

plt.figure()
plt.imshow(chDet.fgIm);

plt.figure()
plt.imshow(chDet.labelI);

print("Close all windows to end test.")
plt.show()


#
#============================== diff01simple =============================
