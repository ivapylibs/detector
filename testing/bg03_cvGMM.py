#!/usr/bin/python
# ============================ image08_bgGMM_cv ==============================
"""
@file       bg03_cvGMM.py
@ingroup    Detector_Testing
    @brief:         Experiment with the Opencv GMM-based background substraction methods.
                    It tests the Opencv's Shadow detection algorithm trained on 
                    training frames with the shadow

    @author:    Yiye        yychen2019@gatech.edu
    @date:      07/23/2021

"""
# ============================ image08_bgGMM_cv ==============================

# ====== [1] setup the environment. Read the data
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/BG')

bg_pure = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_human_wave.avi'))
bg_test_files = []
for i in range(5):
    test_file = os.path.join(dPath, "bgTest_human_puzzle_{}.png".format(i))
    bg_test_files.append(test_file)

# ==== [2] Prepare the bg modeler
bg_params = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=30.,
    detectShadows=True,
    ShadowThreshold=0.55,
)
bg_extractor = BG.bgmodelGMM_cv(params=bg_params)

# test the set and get
hist = 400
bg_extractor.set("History", hist)
assert bg_extractor.get("History") == hist
print("The set and the get function are tested. They are functional")

# ==== [3] Learn the GMM parameters
bg_extractor.doAdapt = True
ret=True
while(bg_pure.isOpened() and ret):
    ret, frame = bg_pure.read()
    if ret:
        bg_extractor.process(frame)

bg_img = bg_extractor.getBackgroundImg()
plt.figure()
plt.title("The background image")
plt.imshow(bg_img)

# ==== [4] Test on the test data
bg_extractor.doAdapt = False
ret=True
for test_file in bg_test_files:
    test_img = cv2.imread(test_file)[:,:,::-1]
    bg_extractor.process(test_img)
    fgMask = bg_extractor.getForeground()
    detResult = bg_extractor.getDetectResult() 
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[0].set_title("The test image")
    axes[1].imshow(detResult, cmap='gray') 
    axes[1].set_title("The detected Foreground(white) and Shadow(gray)")
    axes[2].imshow(fgMask, cmap='gray')
    axes[2].set_title("The foreground")

# TODO: occlude the human hand mask

plt.show()
