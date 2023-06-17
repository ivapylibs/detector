#!/usr/bin/python3
#============================== tgm01basic =============================
'''!
 @brief Simple test of single Gaussian foreground modeling using random data.

'''
#============================== tgm01basic =============================

#
# @file     tgm01basic.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/09          [created]
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#============================== tgm01basic =============================

import numpy as np
import detector.fgmodel.Gaussian as SGM 
import cv2

import camera.utils.display as display

mv = 15.0;
bgModP = SGM.SGMdebug(mu = np.array([mv]), sigma = np.array([10.0]))
bgConf = SGM.CfgSGT.builtForLearning()
bgConf.minSigma = 5;

fgModel = SGM.Gaussian( bgConf, None, bgModP )

for ii in range(210):
  I = np.random.normal(0, 1, (50, 50))
  I[11:18,11:18] = I[11:18,11:18]  + mv;

  fgModel.process(I)

  print("---------------------------------------------")
  print("Measuement top-left), Mean, Variance of model: ",
        I[0,0], fgModel.mu, fgModel.sigma[0])
  #print("Error, Squared Error, errMax", fgModel.errI[0][0], fgModel.sqeI[0][0]/fgModel.config.alpha, fgModel.maxE[0][0])
  print("Max error and threshold are: ", np.amax(fgModel.maxE), fgModel.config.tauSigma, np.amax(fgModel.nrmE))

  print("Counting foreground pixels.", 
        np.count_nonzero(fgModel.fgI))
  #I = np.ones((50,50))

print("Creating another input in another location.");
I = np.random.normal(0,1,(50,50))
I[15:25,30:45] = 15.0
fgModel.process(I)

print("Counting foreground pixels.", 
      np.count_nonzero(fgModel.fgI))

bgs = fgModel.getState()
#print(bgs.bgIm[10:30,25:45])

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
fgIm = cv2.cvtColor(bgs.fgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
display.display_rgb_cv(fgIm, ratio=1)
cv2.waitKey();

#
#============================== tgm01basic =============================
