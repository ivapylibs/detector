#!/usr/bin/python3
#============================== tgm02glove =============================
'''!
@brief Simple test of single Gaussian foreground modeling using glove image.

This is a quick test to figure out the color of the red glove and to
debug the Gaussian foreground model class.  Establishing the red glove
color involved manual selection of the initial mean vector based on
a typical brick red color from a color picker interface and setting
high variance.  The mean and variance was then tuned based on the
output mean and variance when runnning this script.  The values 
coded here in the final version reflect the results from a few test
and update iterations.

While it does a good job catching the glove, the final results might be
a little bit permissive and capture too much.  For Puzzlebot, the idea
is to also incorporate depth information to remove false positives
from the puzzle pieces.

Execution:
----------
Running this script loads the stored image and then processes it.  There
is nothing required. The ''data'' directory has a second glove image
in it. Replacing the "1" in the filename with a "2" will apply foreground
segmentation to that image instead. Both work decently.

'''
#============================== tgm02glove =============================
#
# @file     tgm02basic.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/09          [created]
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#============================== tgm02glove =============================

import numpy as np
import detector.fgmodel.Gaussian as SGM 
import cv2

import camera.utils.display as display

#==[1] Create an instance of a single Gaussian foreground detector
#       
fgModP = SGM.SGMdebug(mu = np.array([130.0,10.0,50.0]), 
                      sigma = 2*np.array([650.0,150.0,250.0]) )
fgModel = SGM.fgGaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )


#==[2] Apply to image data and display outcomes.  Prints the mean and sigma
#       values. Displays the segmentation.
#
Itest = cv2.imread("../data/glove_1.png")[:, :, ::-1]
fgModel.process(Itest)

fgs = fgModel.getState()

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
display.binary_cv(fgs.fgIm, ratio=1)
cv2.waitKey();

#
#============================== tgm02glove =============================
