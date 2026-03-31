from detector.fgmodel.targetMagenta import targetMagenta
import numpy as np
from detector.fgmodel.appearance import fgAppearance
import cv2

# Struct for tModel
class TModel(object):
    def __init__(self, tau=None):
        self.threshold = tau  

class targetDot(fgAppearance):
    '''!
    @ingroup    Detector
    @brief      Class instance for extracting a dot from 
                the image which is all black, except for a white square
                with a dot in the center. The dot is meant to be
                any color distinct from black, white.
    '''

    # ============================= targetDot ============================
    #
    # @brief  Constructor for dot target FG detector.
    #
    # @param[in]  appMod  The model or parameters for the appearance detector.
    #
    def __init__(self, appMod):
        super(targetDot, self).__init__(appMod, None)

    # ============================== measure ==============================
    #
    # @brief  Apply the appearance detector to an image.
    #
    def measure(self, I):

        if self.processor:
            pI = self.processor.apply(I)
        else:
            pI = I

       # Segment out the white region using thresholding 
       
        gray = cv2.cvtColor(pI, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, self._appMod.threshold, 255, cv2.THRESH_BINARY)

        mask_filled = thresh.copy()

        # 2. Fill holes using findContours and drawContours
        contours, hierarchy = cv2.findContours(mask_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            # External contours are the piece; internal are the holes
            # We fill everything white to create a solid object
            cv2.drawContours(mask_filled, contours, i, 255, -1)

        
        # 3. Subtract original mask from filled mask to get just the hole
        just_the_hole = cv2.subtract(mask_filled, thresh)

        kernel = np.ones((3,3), np.uint8)
        just_the_hole = cv2.morphologyEx(just_the_hole, cv2.MORPH_OPEN, kernel)

        # cv2.imshow("hole", just_the_hole)
        # cv2.waitKey(0)

        self.fgIm = just_the_hole.astype(bool)

    def calib(tau):
        tModel= TModel(tau=tau)
        return tModel


    def build_model(threshold):
        tModel = targetDot.calib(threshold)
        magentaDet = targetDot(tModel)
        return magentaDet
    
    

    
#
#=========================== fgmodel/targetDot ==========================

