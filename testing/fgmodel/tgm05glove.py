#!/usr/bin/python3
#============================== tgm05glove =============================
'''!
@brief  Test of single Gaussian foreground model loading applied to glove image.

Follows up on tgm02glove by taking the model, instantiating a Gaussian model,
saving it to file, reloading, and applying.  It should work the same as
the tgm02glove script given that the save/load operations operate result
in the instantiation of a single Gaussian foreground model that is exactly
the same as in the original tgm02glove script.

Execution:
----------
Just run the script.  Nothing is needed from the user. The data source directory
has a second glove image if a change of input is desired.
'''
#============================== tgm05glove =============================
#
# @file     tgm05glove.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/09          [created]
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#============================== tgm05glove =============================

import numpy as np
import detector.fgmodel.Gaussian as SGM 
import cv2

import camera.utils.display as display

#==[1] Create an instance of a single Gaussian foreground detector
#       and save its configuration to a YAML file. Clear out the
#       instance by wiping from variable.
fgModP = SGM.SGMdebug(mu = np.array([130.0,10.0,50.0]), 
                      sigma = 2*np.array([650.0,150.0,250.0]) )
fgModel = SGM.fgGaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )
fgModel.saveConfig('./tgm05glove.yaml')

fgModel = None

#==[2] Load the saved configuration and instantiate detector again.
#       
fgModel = SGM.fgGaussian.loadFromConfig('./tgm05glove.yaml')

#==[3] Apply to image data and display outcomes.  Prints the mean and sigma
#       values. Displays the segmentation.
#
Itest = cv2.imread("../data/glove_1.png")[:, :, ::-1]
fgModel.process(Itest)
fgs = fgModel.getState()

print('Mean  is:' , fgModel.mu)
print('Var   is:' , fgModel.sigma)
print('For comparison, the original tgm02glove outcomes were')
print('Mean was: [128.1230241   11.65957119  48.54421244]')
print('Var  was: [1183.82761651  273.89911195  457.07797222]');

display.binary_cv(fgs.fgIm, ratio=1)
cv2.waitKey();

#==[4] Done with test script. Return to command line.

#
#============================== tgm05glove =============================
