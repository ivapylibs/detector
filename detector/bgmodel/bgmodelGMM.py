# ============================== bgmodelGMM ==============================
"""
  function bgh = bgmodelGMM(mu, sigma, parms)


  Implements a background modeling foreground detector using a Gaussian 
  mixture model.  This model involves two parts, one is an estimator,
  and the other is a change detector.  The estimator is a static 
  prediction observer on the mean and variance with fixed udpate gains.  
  The change detector is threshold-based.  Estimate corrections rely
  on the change detector.

  Inputs:
    mu 		- the means of the Gaussian models.
    sigma 		- the variance of the Gaussian models.
    weights	- the weights of the Gaussian models.
    parms		- [optional] structure with parameters specified.

  Fields of the parms structure:
    sigma		- Initial variance to use if sigma is empty.
    thresh		- Threshold for determining foreground.
    alpha		- Update rate for mean and variance.
    rho		- Update rate for weights.
    cnctd_thrsh	- Connected component threshold size for false blob removal.
    se3		- structuring element for morphological operations. 
    improcessor	- Image processor interface object for performing
					  pre-, mid-, and post-processing of signal.

		A note on the improcessor.  If the basic version is used, then
		it performs pre-processing. If a triple version is used, then
		the mid-processor will perform operations on the detected part
		rather than the default operations.  The mid-processor can be used
		to test out different options for cleaning up the binary data.

  Output:
    bgh		- handle to interface object.
"""
# =============================== bgmodelGMM ==============================
"""

  Name:		bgmodelGMM.m

  Author:		Gbolabo Ogunmakin, 		gogunmakin3@gatech.edu
				Patricio A. Vela,		pvela@gatech.edu
                Yiye Chen (py),         yychen2019@gatech.edu

  Created: 	2011/01/31 		(original: bg_model.m)
  Modified:	2013/01/20 
  Translated to python: 2021/07/22

  Notes:
    set tabstop = 4, indent = 2.
"""
#=============================== bgmodelGMM ==============================

from detector.inImage import inImage
import numpy as np
import cv2

class bgmodelGMM(inImage):
    def __init__(self, K, **kwargs):
        self.bgSubstractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=100,
            detectShadows=True # it doesn't seem to be helpful
        )
        #self.bgSubstractor=cv2.createBackgroundSubtractorMOG
        super().__init__()

        self.fgI = None
        if 'adapt_rate' in kwargs.keys():
            self.adapt_rate = kwargs['adapt_rate']
        else:
            self.adapt_rate = -1 # The algorithm will automatically determine the lr

        self.doAdapt = True

    def measure(self, I):
        """
        Apply the GMM and get the fgI mask
        """
        #self.fgI = np.zeros_like(I[:, :, 0], dtype=bool)
        #return None 
        if self.doAdapt:
            self.fgI = self.bgSubstractor.apply(I, learningRate=self.adapt_rate)
        else:
            self.fgI = self.bgSubstractor.apply(I, learningRate=0)

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
