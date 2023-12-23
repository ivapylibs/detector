#!/usr/bin/python3
#============================== sgm01basic =============================
'''!
 @brief Simple test of single Gaussian background modeling using random data.

'''
#============================== sgm01basic =============================

#
# @file     sgm01basic.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/09          [created]
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#============================== sgm01basic =============================

import numpy as np
import detector.bgmodel.Gaussian as SGM 
import cv2

import camera.utils.display as display


bgModel = SGM.bgGaussian( SGM.CfgSGM.builtForLearning() )

for ii in range(210):
  I = np.random.normal(0, 1, (50, 50))
  #print(I[0:4,0:4])

  bgModel.process(I)

  print("---------------------------------------------")
  print("Measuement, Mean, Variance of top-left pixel: ",
        I[0,0], bgModel.mu[0][0], bgModel.sigma[0][0])
  print("Error, Squared Error, errMax", bgModel.errI[0][0], bgModel.sqeI[0][0]/bgModel.config.alpha, bgModel.maxE[0][0])
  print("Max error and threshold are: ", np.amax(bgModel.maxE), bgModel.config.tauSigma, np.amax(bgModel.nrmE))

  print("Counting foreground pixels.", 
        np.count_nonzero(bgModel.bgI))
  #I = np.ones((50,50))

print("Creating an egregiously wrong input.");
I = np.random.normal(0,1,(50,50))
I[15:25,30:45] = 200.0
bgModel.process(I)

print("Counting foreground pixels.", 
      np.count_nonzero(bgModel.bgI))

bgs = bgModel.getState()
#print(bgs.bgIm[10:30,25:45])

bgIm = cv2.cvtColor(bgs.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
display.display_rgb_cv(bgIm, ratio=1)
cv2.waitKey();

#
#============================== sgm01basic =============================
