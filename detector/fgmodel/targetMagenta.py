import numpy as np
from detector.fgmodel.appearance import appearance

# Struct for tModel
class TModel(object):
    def __init__(self, R=None, T=None, tau=None, classify=None, vectorize=None):
        self.R = R
        self.T = T
        self.tau = tau
        self.classify = classify
        self.vectorize = vectorize

class targetMagenta(appearance):

    # ============================= targetMagenta ============================
    #
    # @brief  Constructor for magenta target FG detector.
    #
    # @param[in]  appMod  The model or parameters for the appearance detector.
    #
    def __init__(self, appMod):
        super(targetMagenta, self).__init__(appMod, None)

    # ============================== measure ==============================
    #
    # @brief  Apply the appearance detector to an image.
    #
    def measure(self, I):

        if self.processor:
            pI = self.processor.apply(I)
        else:
            pI = I

        if self._appMod.vectorize:
            imDat = np.array(pI).reshape(-1,pI.shape[2]).T
            fgB = self._appMod.classify(imDat)
            self.fgIm = np.array(fgB).reshape(pI.shape[0], pI.shape[1])
        else:
            self.fgIm = self._appMod.classify(pI)


    def calib(tau):
        tModel= TModel()
        M = np.array([1, -2, 1])
        tModel.R = M/np.linalg.norm(M)
        tModel.T = 0
        tModel.tau = tau
        tModel.classify = lambda c: tModel.R @ c + tModel.T > tModel.tau
        tModel.vectorize = True
        return tModel


    def build_model(threshold):
        tModel = targetMagenta.calib(threshold)
        magentaDet = targetMagenta(tModel)
        return magentaDet
    
    

    
#
#=========================== fgmodel/targetMagenta ==========================

