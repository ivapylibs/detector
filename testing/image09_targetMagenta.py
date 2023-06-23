#!/usr/bin/python
#=========================== image09_targetMagenta ===========================
"""!
@brief  Test the Magenta foreground target detector. 

"""
#=========================== image09_targetMagenta ===========================
#
#
# @author   Yiye Chen        yychen2019@gatech.edu
# @date     2021/07/16
#
#=========================== image09_targetMagenta ===========================

import cv2
import numpy as np
from detector.fgmodel.targetMagenta import targetMagenta

if __name__=='__main__':
    img_test = cv2.imread('data/img.png')[:, :, ::-1]

    magentaDetector = targetMagenta.build_model(25)

    # ======= [3] test on the test image and show the result
    magentaDetector.process(img_test)
    fgmask = magentaDetector.getForeGround()
    cv2.imshow("The test image", img_test[:, :, ::-1])
    cv2.imshow("The FG detection result", fgmask.astype(np.uint8)*255)

    print("Press any key to exit:")
    cv2.waitKey()

#
#=========================== image09_targetMagenta ===========================
