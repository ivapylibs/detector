#!/usr/bin/python

import cv2
import numpy as np
import detector.bgmodel.inCorner as bgdet

if __name__=='__main__':
    #filename = 'data/workspace_no_pieces_Color.png'
    filename = 'data/workspace_with_pieces_dinosaur_02.png'
    #filename = 'data/workspace_arm_no_pieces_Color_Color.png'
    #filename = 'data/workspace_with_arm_pieces_dinosaur_Color_Color.png'
    #filename = 'data/img.png'
    img_test = cv2.imread(filename)

    inShad = 0
    tRad = 80 - 20*inShad      # No shadows = 100. In shadows = 80.
    bgModel = bgdet.inCorner.build_spherical_blackBG(tRad^2, 0)
    bgDetector = bgdet.inCorner()
    bgDetector.set_model(bgModel)

    # ======= [3] test on the test image and show the result
    bgDetector.process(img_test)
    bgmask = bgDetector.getState()
    cv2.imshow("The test image", img_test)
    cv2.imshow("The FG detection result", 255-bgmask.x.astype(np.uint8)*255)

    bgM = cv2.cvtColor(255-bgmask.x.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    fgI = cv2.bitwise_and(img_test, bgM)

    cv2.imshow("Masked Image", fgI)

    print("Press any key to exit:")
    cv2.waitKey()
