import numpy as np
from detector.fgmodel.appearance import fgAppearance

# Struct for tModel
class TModel(object):
    def __init__(self, R=None, T=None, tau=None, classify=None, vectorize=None):
        self.R = R
        self.T = T
        self.tau = tau
        self.classify = classify
        self.vectorize = vectorize

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

        im = np.asarray(pI)

        # Convert RGB image to grayscale. If already single-channel, keep as is.
        if im.ndim == 3 and im.shape[2] >= 3:
            gray = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
        elif im.ndim == 2:
            gray = im
        else:
            raise ValueError("Unsupported image shape for targetDot.measure")

        gray = np.asarray(gray, dtype=np.float32)
        if gray.max() <= 1.0:
            gray = gray * 255.0

        # False near black/white ends, True in the middle band.
        self.fgIm = self._appMod.classify(gray)


    def calib(tau):
        tModel= TModel()
        tModel.R = None
        tModel.T = None

        # tau is tolerance from both grayscale spectrum ends [0, 255].
        # Pixels in [0, tau] and [255 - tau, 255] are background (False).
        # Pixels in (tau, 255 - tau) are foreground (True).
        tModel.tau = float(np.clip(tau, 0.0, 127.0))
        tModel.classify = lambda g: np.logical_and(g > tModel.tau, g < (255.0 - tModel.tau))
        tModel.vectorize = False
        return tModel


    def build_model(threshold):
        tModel = targetDot.calib(threshold)
        dotDet = targetDot(tModel)
        return dotDet
    
    

    
#
#=========================== fgmodel/targetDot ==========================

