#=============================== bgmodelGMM ==============================
"""
  @class bgmodelGMM

  bgmodelGMM(mu, sigma, parms)

  Implements a background modeling foreground detector using a Gaussian 
  mixture model.  This model involves two parts, one is an estimator,
  and the other is a change detector.  The estimator is a static 
  prediction observer on the mean and variance with fixed udpate gains.  
  The change detector is threshold-based.  Estimate corrections rely
  on the change detector.

  Inputs:
    mu          - the means of the Gaussian models.
    sigma       - the variance of the Gaussian models.
    weights     - the weights of the Gaussian models.
    parms       - [optional] structure with parameters specified.

  Fields of the parms structure:
    sigma       - Initial variance to use if sigma is empty.
    thresh      - Threshold for determining foreground.
    alpha       - Update rate for mean and variance.
    rho         - Update rate for weights.
    cnctd_thrsh - Connected component threshold size for false blob removal.
    se3	        - structuring element for morphological operations. 
    improcessor - Image processor interface object for performing
					  pre-, mid-, and post-processing of signal.

		A note on the improcessor.  If the basic version is used, then
		it performs pre-processing. If a triple version is used, then
		the mid-processor will perform operations on the detected part
		rather than the default operations.  The mid-processor can be used
		to test out different options for cleaning up the binary data.

"""
#=============================== bgmodelGMM ==============================
"""

  Name:		    bgmodelGMM.m

  Author:		Gbolabo Ogunmakin, 		gogunmakin3@gatech.edu
				Patricio A. Vela,		pvela@gatech.edu
                Yiye Chen (py),         yychen2019@gatech.edu

  Created: 	    2011/01/31 		(original: bg_model.m)
  Modified:	    2013/01/20 
  Translated to python: 2021/07/22

  Notes:
    set tabstop = 4, indent = 2.
"""
#=============================== bgmodelGMM ==============================

import numpy as np
import cv2
from dataclasses import dataclass

from detector.inImage import inImage

class bgmodelGMM(inImage):
    """
    Translation of the ivaMatlib/bgmodelGMM
    NOTE: Not implemented.
    """
    def __init__(self, K, **kwargs):
        super().__init__()

        self.fgI = None

        self.doAdapt = True

    def measure(self, I):
        """
        Apply the GMM and get the fgI mask
        """
        pass

    def compProbs(self):
        return None
    
    def correct(self, fg):
        """
        Update the existing model parameters
        """
        return None 
    
    def adapt(self):
        """
        Create new model
        """
        return None

    def detectFG(self):
        """
        Apply post-process to the fg mask?
        """
        return None 

    def process(self, img):
        self.measure(img)
        return None

    def set(self, fname, fval):
        return None
    
    def get(self, fname):
        return None 

    def getstate(self):
        return None 
    
    def getForeground(self):
        """
        Get the current foreground estimate
        """
        return self.fgI
    
    def getProbs(self):
        """
        Get the current probability and the generating model
        """
        prI = None
        idI = None
        return prI, idI

#============================= bgmodelGMM_cv =============================
"""
  @class bgmodelGMM_cv

  Wrapper for the OpenCV Gaussian mixture model foreground detection
  implementation.  This model involves two parts, one is an estimator,
  and the other is a change detector.  The estimator is a static
  prediction observer on the mean and variance with fixed udpate gains.
  The change detector is threshold-based.  

"""
#=============================== bgmodelGMM ==============================

@dataclass
class Params_cv:
    """
    The parameters for the bgmodelGMM_cv

    For mored detail on the parameters, check OpenCV documentation.
    @param  history         @< Number of frames to use for model estimation
    @param  NMixtures       @< Number of mixtures
    @param  varThreshold    @< Variance threshold for detection.
    @param  detectShadow    @< Boolean: detect shadows or not.
    @param  adapt_rate      @< Adaptation rate.
    """
    history:    int = 300
    NMixtures:  int = 5
    varThreshold:       float = 50.
    detectShadows:      bool = True
    ShadowThreshold:    float = 0.5
    adapt_rate:         float = -1      # will automatically choose rate

class bgmodelGMM_cv(inImage):
    """
    GMM Background Substraction method MOG2 based on the OpenCV.
    Include shadow detection feature.

    The detection algorithm will first use the GMM to detect a potential
    foreground mask, each pixel of which will be checked for the color
    distort and the intensity decay. 
    """
    def __init__(self, params: Params_cv):

        super().__init__()
        # The shadow detection methods from the following paper:
        # http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
        # The source code for shadow detection:
        # https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp#L477-L520
        self.bgSubstractor = cv2.createBackgroundSubtractorMOG2(
            history=params.history,
            varThreshold=params.varThreshold,
            detectShadows=params.detectShadows 
            # Shadow detection doesn't seem to be helpful.
        )
        self.set("ShadowThreshold", params.ShadowThreshold)
        self.set("NMixtures", params.NMixtures)

        # parameters for apply
        self.doAdapt = False    # default false. Set to True during calibration
        self.adapt_rate = params.adapt_rate

        # for storing the result
        self.detResult      = None
        self.shadow_mask    = None
        self.fg_mask        = None

    def measure(self, I):
        """
        Apply the GMM and get the fgI mask
        """
        pass

    def correct(self, fg):
        """
        Update the existing model parameters
        """
        pass
    
    def adapt(self):
        """
        Create new model
        """
        pass

    def detectFG(self):
        """
        Apply post-process to the fg mask?
        """
        pass

    def process(self, img):
        # It seems that the opencv's apply will do everything in the process
        if self.doAdapt:
            self.detResult = self.bgSubstractor.apply(img, learningRate=self.adapt_rate)
            self.fg_mask = (self.detResult == 255)
            self.shadow_mask = (self.detResult == 127)
        else:
            self.detResult = self.bgSubstractor.apply(img, learningRate=0)
            self.fg_mask = (self.detResult == 255)
            self.shadow_mask = (self.detResult == 127)

    def set(self, fname, fval):
        """
        Check the following link for what you can set:
        https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#acdb85152b349d70561fecc4554ad76e6

        @param[in]  fname           The name of the parameter to set. will invoke set+fname for the opencv MOG2
        @param[in]  fval            The value to set

        example:

        det = bgmodelGMM_cv()
        det.set(History, 200)   # will invode setHistory(200) function from the link
        """
        eval(
            "self.bgSubstractor.set"+fname+"(" + str(fval) + ")"
        )

    
    def get(self, fname):
        """
        Check the following link for what you can set:
        https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#acdb85152b349d70561fecc4554ad76e6

        @param[in]  fname           The name of the parameter to set. will invoke set+fname for the opencv MOG2

        example:

        det = bgmodelGMM_cv()
        det.get(History)   # will invode getHistory() function from the link
        """

        fval = eval(
            "self.bgSubstractor.get"+fname+"()"
        )
        return fval

    def getstate(self):
        pass

    def getDetectResult(self):
        """
        Get the detection result, including the foreground and the shadow
        The foreground pixels' value will be 255, wherease the shadow pixels' will be 127. So the shadows will look like shadow(darker)
        """
        return self.detResult
    
    def getForeground(self):
        """
        Get the current foreground estimate
        """
        return self.fg_mask
    
    def getBackground(self):
        """
        Get the current background mask
        """
        return ~self.fg_mask

    def getShadow(self):
        """
        Get the detected shadow mask
        """
        return self.shadow_mask

    
    def getProbs(self):
        """
        Get the current probability and the generating model
        """
        pass
        prI = None
        idI = None
        return prI, idI
    
    def getBackgroundImg(self):
        """
        Get the background image in RGB, based on the background GMM model.
        The image will be the weighted mean of the Gaussians' means, according to the source code:
        https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp#L887-L927

        Returns:
            bgImg (H, W, 3):    The back
        """
        return self.bgSubstractor.getBackgroundImage()[:,:,::-1]
