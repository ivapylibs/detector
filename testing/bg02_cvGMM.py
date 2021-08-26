#!/usr/bin/python3
# ============================ image08_bgGMM_cv ==============================
"""
    @brief:         Experiment with the Opencv GMM-based background substraction methods.

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

bg_pure = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_empty_table.avi'))
bg_test_files = []
for i in range(5):
    test_file = os.path.join(dPath, "bgTest_human_puzzle_{}.png".format(i))
    bg_test_files.append(test_file)

# ==== [2] Prepare the bg modeler
bg_params = BG.Params_cv()
bg_extractor = BG.bgmodelGMM_cv(params=bg_params)

# test the set and get
hist = 400
bg_extractor.set("History", hist)
assert bg_extractor.get("History") == hist
print("The set and the get function are tested. They are functional")

# ==== [3] Learn the parameters
bg_extractor.doAdapt = True
ret=True
while(bg_pure.isOpened() and ret):
    ret, frame = bg_pure.read()
    if ret:
        bg_extractor.process(frame)

# ==== [4] Test on the test data
bg_extractor.doAdapt = False
ret=True
for test_file in bg_test_files:
    test_img = cv2.imread(test_file)[:,:,::-1]
    bg_extractor.process(test_img)
    fgMask = bg_extractor.getForeground()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[1].imshow(test_img) #TODO: opencv bg substractor can provide shadow detection result. show it
    axes[2].imshow(fgMask, cmap='gray')

plt.show()