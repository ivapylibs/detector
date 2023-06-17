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


Itest = cv2.imread("../data/glove_1.png")[:, :, ::-1]

fgModP = SGM.SGMdebug(mu = np.array([130.0,10.0,50.0]), 
                      sigma = 2*np.array([650.0,150.0,250.0]) )

fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

fgModel.process(Itest)

#  print("---------------------------------------------")
#  print("Measuement top-left), Mean, Variance of model: ",
#        I[0,0], fgModel.mu, fgModel.sigma[0])
#  #print("Error, Squared Error, errMax", fgModel.errI[0][0], fgModel.sqeI[0][0]/fgModel.config.alpha, fgModel.maxE[0][0])
#  print("Max error and threshold are: ", np.amax(fgModel.maxE), fgModel.config.tauSigma, np.amax(fgModel.nrmE))
#
#  print("Counting foreground pixels.", 
#        np.count_nonzero(fgModel.fgI))
#  #I = np.ones((50,50))

fgs = fgModel.getState()
#print(bgs.bgIm[10:30,25:45])

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
fgIm = cv2.cvtColor(fgs.fgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
display.display_rgb_cv(fgIm, ratio=1)
cv2.waitKey();

#
#============================== tgm02glove =============================
