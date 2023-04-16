#!/usr/bin/python3

import cv2
import numpy as np
import detector.bgmodel.inCorner as bgdet

if __name__=='__main__':
    #filename = 'data/workspace_no_pieces_Color.png'
    #filename = 'data/workspace_with_pieces_dinosaur_Color.png'
    #filename = 'workspace_arm_no_pieces_Color_Color.png'
    filename = 'data/workspace_with_arm_pieces_dinosaur_Color_Color.png'
    #filename = 'data/img.png'
    img_test = cv2.imread(filename)

    bgModel = bgdet.inCorner.build_model_blackBG(-93, 0)
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
