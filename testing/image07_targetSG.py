#!/usr/bin/python
#============================== image07_targetSG =============================
## @file
# @brief  Test the single-Gaussian-color-modeling-based foreground detector. 
# 
# Grabs data from saved images and applies.
#
# @author   Yiye Chen        yychen2019@gatech.edu
# @date     2021/07/16
# @ingroup  TestDetector
# @quitf
#
#============================== image07_targetSG =============================

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
from detector.fgmodel.targetSG import targetSG

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data/FG_glove')

img_train = cv2.imread(
    os.path.join(dPath, "glove_1.png")
)[:, :, ::-1]


img_test = cv2.imread(
    os.path.join(dPath, "glove_2.png")
)[:, :, ::-1]

# ======= [2] build the detector instance
SGDetector = targetSG.buildFromImage(img_train)

# ======= [3] test on teh test image and show the result
SGDetector.process(img_test)
fgmask = SGDetector.getForeGround()
cv2.imshow("The test image", img_test[:, :, ::-1])
cv2.imshow("The FG detection result", fgmask.astype(np.uint8)*255)

print("Press any key to exit:")
cv2.waitKey()

#
#============================== image07_targetSG =============================
