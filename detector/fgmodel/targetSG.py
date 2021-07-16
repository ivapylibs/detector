# =========================== fgmodel/targetSG ==========================
"""
 @class    fgmodel.targetSG

 @brief    Interfaces for the target detection module 
            based on the single Gaussian RGB color modeling


 The target pixels are assumed to have similar RGB value, and is modeled as a
 single Guassian distribution.  It then transform the
 distribution into a Gaussian with uncorrelated components by
 diagonalizing the covariance matrix. The fg detection is done by
 thresholding each component independently in the transformed color space(tilt):
       |color_tilt_i - mu_tilt_i| < tau * cov_tilt_i, for all i=1,2,3
 tau can be user-input

 The interface is adapted from the fgmodel/targetNeon.
"""
# =========================== fgmodel/targetSG ==========================
"""
 @file     targetSG.m

 @author  Yiye Chen,       yychen2019@gatech.edu
 @date     2021/02/28 [Matlab version]
           2021/07/15 [Python version]

 @classf   fgmodel
"""
# =========================== fgmodel/targetSG ==========================

from detector.fgmodel.appearance import appearance
import numpy as np


class tModel_SG():
    """
    The target model class storing the statistics of the target color
    """
    def __init__(self):
        self.mu = None
        self.sig = None
        self.sigEignVec = None

        self.has_Diag = False

    def getCovEign(self):
        self.has_diag = True
        return None
    

class targetSG(appearance):
    """
    The single-Gaussian based target detection class.
    The target color is modeled as a single-Guassian distribution
    """
    def __init__(self):
        super().__init__(None, None)
        self.foo = None
    
    def measure(self, I):
        self.fgIm = np.ones_like(I, dtype=np.bool)
        return None

    def saveMod(self, filename):
        return None
    
    def loadMod(self, filename):
        return None

    @staticmethod
    def _calibSimple():
        """
        calibrate the target model given a set of target pixels 
        """
        return None
    
    @staticmethod
    def buildSimple():
        """
        Return a targetSG detector instance given a set of target pixels
        """
        return None
    
    @staticmethod
    def _calibImage(img):
        """
        calibrate and return the target model given an image
        """
        return None

    @staticmethod 
    def buildImage(img):
        """
        Return a target SG detector instance given an image.
        The user will be asked to select the target area from teh image
        """
        det = targetSG()
        return det 
