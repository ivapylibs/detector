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


Itest = cv2.imread("../data/glove_1.png")[:, :, ::-1]

fgModP = SGM.SGMdebug(mu = np.array([135.0,20.0,10.0]), 
                      sigma = np.array([500.0,500.0,500.0]) )

fgModel = SGM.Gaussian( SGM.CfgSGT.builtForLearning(), None, fgModP )

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
#============================== tgm01basic =============================
