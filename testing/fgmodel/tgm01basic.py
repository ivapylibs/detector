#!/usr/bin/python3
#============================== tgm01basic =============================
'''!
 @brief Simple test of single Gaussian foreground modeling using random data.

Defines the mean and sigma value using hard coded quantities and generates
random matrix data consistent with the values.  The random matrix created
consists of not foreground values to which a submatrix is replaced with
foreground values.   

Generally exists to test out the implementation and debug silly syntactical
problems (due to me learning python as I go).

Execution:
----------
Just execute. No user input needed until end to close window and quit.
Will display outcome. Should be a small black image with a small white
square in it.  The square is the introduced foreground data within the
"background" image generated from random dagta.

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

#==[1] Create the foreground detector instance. Set parameters to
#       reflect the random data to generate.
#
mv = 15.0;
bgModP = SGM.SGMdebug(mu = np.array([mv]), sigma = np.array([10.0]))
bgConf = SGM.CfgSGT.builtForLearning()
bgConf.minSigma = 5;

fgModel = SGM.Gaussian( bgConf, None, bgModP )

#==[2] Run a bunch of times as determined by loop range.
#       Will continuously output processing results.
#       Not displayed to a window.
for ii in range(210):
  I = np.random.normal(0, 1, (50, 50))
  I[11:18,11:18] = I[11:18,11:18]  + mv;

  fgModel.process(I)

  print("---------------------------------------------")
  print("Measuement top-left), Mean, Variance of model: ",
        I[0,0], fgModel.mu, fgModel.sigma[0])
  print("Max error and threshold are: ", np.amax(fgModel.maxE), 
                                         fgModel.config.tauSigma, 
                                         np.amax(fgModel.nrmE))
  print("Counting foreground pixels.", np.count_nonzero(fgModel.fgI))


#==[3] Once loop is over create another test image with the target in a 
#       "novel" location. Print and display outcomes.  The foreground
#       pixel counting is a debugging step since the target size
#       is known. During the loop is has an area of 49 pixels (7x7).
#       In the final test it has an area of 150 pixels (10x15).
#
print("Creating another input in another location.");
I = np.random.normal(0,1,(50,50))
I[15:25,30:45] = 15.0
fgModel.process(I)

print("Counting foreground pixels.", np.count_nonzero(fgModel.fgI))

bgs = fgModel.getState()
print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)

display.binary_cv(bgs.fgIm, ratio=1)
cv2.waitKey();

#
#============================== tgm01basic =============================
