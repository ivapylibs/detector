#===================== detector.fgmodel.targetCorner =====================
'''!
  @brief    Class instance for extracting targets that lie with the corner
            region of the RGB color cube, or the extremal regions of a
            color-space based on a planar cut.

'''
#===================== detector.fgmodel.targetCorner =====================
'''!

  @author   Patricio A. Vela,       pvela@gatech.edu
  @date     2023/04/13


'''
# NOTE:
#   Expects 100 character width due to 4-space indents in python.
#
#===================== detector.fgmodel.targetCorner =====================

import numpy as np
from detector.fgmodel.appearance import appearance

# Struct for tModel
class TModel(object):
    def __init__(self, n=None, d = None, tau=None, classify=None, vectorize=None):
        self.n = n
        self.d = d
        self.tau = tau
        self.classify = classify
        self.vectorize = vectorize

class targetCorner(appearance):

    #=========================== targetCorner ==========================
    #
    # @brief  Constructor for corner color model target FG detector.
    #
    # @param[in]  appMod  The model or parameters for the appearance detector.
    #
    def __init__(self, appMod):
        super(targetCorner, self).__init__(appMod, None)

    #============================= measure =============================
    #
    #
    def measure(self, I):
        '''!
        @brief  Apply the appearance detector to an image.
        '''

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


    #============================== calib ==============================
    #
    def calib(n, d, tau):
        '''!
        @brief  Calibrate the model by setting the distance and threshold parameter(s). 
        '''
        tModel = TModel(n, d, tau)

        if (tModel.d == 0):
          tModel.classify = lambda c: tModel.n @ c > tModel.tau
        else:
          tModel.classify = lambda c: tModel.n @ c + tModel.d > tModel.tau

        tModel.vectorize = True

        return tModel


    #=========================== build_model ===========================
    #
    @staticmethod
    def build_model(threshold):
        tModel      = targetCorner.calib(threshold)
        magentaDet  = targetCorner(tModel)

        return magentaDet

    
#
#===================== detector.fgmodel.targetCorner =====================
