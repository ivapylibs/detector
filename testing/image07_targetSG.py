#!/usr/bin/python3
# ============================ image07_targetSG ==============================
"""
    @brief:         Code to test out the single-Gaussian-color-modeling-based 
                    foreground detector. 
"""
# ============================ image07_targetSG ==============================

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
from detector.fgmodel.targetSG import targetSG

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data')

img_train = cv2.imread(
    os.path.join(dPath, "glove_0.png")
)


img_test = cv2.imread(
    os.path.join(dPath, "glove_0.png")
)

# ======= [2] build the detector instance
SGDetector = targetSG.buildImage(img_train)

# ======= [3] test on teh test image and show the result
SGDetector.process(img_test)
fgmask = SGDetector.getForeGround()
cv2.imshow("the detection result", fgmask.astype(np.uint8)*255)
cv2.waitKey()
