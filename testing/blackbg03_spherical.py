#!/usr/bin/python
#============================ blackbg03_spherical ============================
'''!

@brief  Run inCorner background model detector on image, with a black spherical model.


Execution:
----------
Select file in script if desired.
Just run. It displays masked image.  Black area should be uniformly black except 
at edges.
Hit "q" to quit.

'''
#============================ blackbg03_spherical ============================

#
# @file     blackbg03_spherical.py
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/05/XX
#
# NOTE: Indent set to 4 spaces with conversion. 90-100 columns.
#
#============================ blackbg03_spherical ============================


import cv2
import numpy as np
import detector.bgmodel.inCorner as bgdet

if __name__=='__main__':
    #filename = 'data/workspace_no_pieces_Color.png'
    filename = 'data/workspace_with_pieces_dinosaur_02.png'
    #filename = 'workspace_arm_no_pieces_Color_Color.png'
    #filename = 'data/workspace_with_arm_pieces_dinosaur_Color_Color.png'
    #filename = 'data/img.png'
    img_test = cv2.imread(filename)

    alpha = 10
    cent  = np.array([[1],[1],[1]])*alpha*2
    bgModel = bgdet.SphericalModel.build_model(cent, (45+alpha)^2, 0)
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

#
#============================ blackbg03_spherical ============================
