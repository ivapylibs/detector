#!/usr/bin/python
# ============================ image08_targetSG_imgDiff ==============================
"""
    @brief:         Code to test out the single-Gaussian-color-modeling-based 
                    foreground detector, where the foreground color statistics 
                    is obtained by the image difference method

    @ author:   Yiye        yychen2019@gatech.edu
    @ date:     09/26/2021
"""
# ============================ image08_targetSG_imgDiff ==============================

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import detector.fgmodel.targetSG as targetSG
import improcessor.mask as maskproc

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data/FG_glove')

img_bg_train = cv2.imread(
    os.path.join(dPath, "empty_table_0.png")
)[:,:, ::-1]

img_fg_train = cv2.imread(
    os.path.join(dPath, "calibrate_glove_0.png")
)[:, :, ::-1]


img_test = cv2.imread(
    os.path.join(dPath, "puzzle_human_robot_6.png")
)[:, :, ::-1]

# ======= [2] build the detector instance
fh, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_bg_train)
axes[0].set_title("Input background image")
axes[1].imshow(img_fg_train)
axes[1].set_title("Input foreground image")
axes[2].set_title("The extracted training colors")
SGDetector = targetSG.targetSG.buildImgDiff(img_bg_train, img_fg_train, vis=True, ax=axes[2], 
    params=targetSG.Params(
        det_th=8
    )
)
# will need the processor
processor=maskproc.mask(
    maskproc.mask.getLargestCC, ()
)

# ======= [3] test on teh test image and show the result
SGDetector.process(img_test)
fgmask = SGDetector.getForeGround()
fgmask = processor.apply(fgmask)
fg, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(img_test)
axes[0].set_title("The test image")
axes[1].imshow(fgmask, cmap="gray")
axes[1].set_title("The foreground mask")

plt.show()
