#!/usr/bin/python3
# ============================ image08_bgGMM ==============================
"""
    @brief:         The test file for the background substraction using the 
                    Gaussian Mixture Model.

    @ author:   Yiye        yychen2019@gatech.edu
    @ date:     07/23/2021
"""
# ============================ image08_bgGMM ==============================

# ====== [1] setup the environment. Read the data
import os
import numpy as np
import cv2
from detector.bgmodel.bgmodelGMM import bgmodelGMM

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data')

bg_pure = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_noFg.avi'))
bg_hand = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_Fg.avi'))


# ==== [2] Prepare the bg modeler
bg_extractor = bgmodelGMM(K=1)

# ==== [3] Learn the parameters
bg_extractor.doAdapt = True
ret=True
while(bg_pure.isOpened() and ret):
    ret, frame = bg_pure.read()
    if ret:
        bg_extractor.process(frame)

# ==== [4] Apply
bg_extractor.doAdapt = False
ret=True
while(bg_hand.isOpened(), ret):
    ret, frame = bg_hand.read()
    if ret:
        bg_extractor.process(frame)
        fgMask = bg_extractor.getForeground()
        cv2.imshow("original", frame)
        cv2.imshow("fg", fgMask.astype(np.uint8) * 255)
        if cv2.waitKey(100) & 0xFF == ord('q'):
              break