#============================ detector.fgmodel.inCorner ============================
'''!
  @brief    Class instance for extracting background that lies with the corner
            region of the RGB color cube, or the extremal regions of a
            color-space based on a planar cut.

  The idea is similar to the fgmodel approach based on planar cuts but
  with the presumption that the background is what has the unique color
  profile.  The presumption is that the background lies in a region of 
  color-space that is extremal so that a planar surface separates it from
  the expected foreground colors.
'''
#============================ detector.fgmodel.inCorner ============================
'''!

  @author   Patricio A. Vela,       pvela@gatech.edu
  @date     2023/04/13
'''
# NOTE:
#   Expects 100 character width due to 4-space indents in python.
#   Comment separators also wider to 84 characters rather than 74, with others
#   scaled up too (72 -> 78).
#
#============================ detector.fgmodel.inCorner ============================

import numpy as np
from detector.inImage import inImage

# Struct for tModel
class PlanarModel(object):
    '''!
      @brief    Specification data class that describes the planar cut classifier
                and its functional properties.
    '''
    def __init__(self, n=None, d = None, tau=None, classify=None, vectorize=True):
        self.n = n
        self.d = d
        self.tau = tau
        self.classify = classify
        self.vectorize = vectorize

    @staticmethod
    def build_model(n, d, tau, isVectorized=True):
        '''!
        @brief  Build a model given the arguments as specifications.

        @param[in]  n               Normal vector.
        @param[in]  d               Separating boundary offset.
        @param[in]  tau             Threshold/one-side margin.
        @param[in]  isVectorized    Boolean: operation should be vectorized or not.
                                    Default is True.
        '''

        theModel = PlanarModel(n, d, tau)

        if (theModel.d == 0):
          theModel.classify = lambda c: theModel.n @ c > theModel.tau
        else:
          theModel.classify = lambda c: theModel.n @ c + theModel.d < theModel.tau

        theModel.vectorize = isVectorized

        return theModel

class inCorner(inImage):

    #================================ inCorner ===============================
    #
    #
    def __init__(self, processor = None, bgMod = None):
        '''!
        @brief  Constructor for corner color model target FG detector.
    
        @param[in]  appMod  The model or parameters for the appearance detector.
        '''
        super(inCorner, self).__init__(processor)

        self.bgModel = bgMod

    #============================= specify_model =============================
    #
    def specify_model(self, n, d, tau, isVectorized = True):
        '''!
        @brief  Specify the model by providing normal, distance, and threshold parameter(s). 

        The distance offset and threshold operate additively, this it may be the case
        the one is set to zero and the other is not.  Alternatively, it may be more
        intuitive to have them be separate since dist defines the location of the seprating
        plane and tau defines the detection margin/slack associated to the separating plane.
        It all depends on what is more intuitive to the designer.  All options permitted in
        this case.

        @param[in]  n       Normal vector.
        @param[in]  d       Distance offset.
        @param[in]  tau     Threshold.
        '''
        tModel = PlanarModel.build_model(n, d, tau, isVectorized)
        self.set_model(tModel)

        return tModel


    #=============================== set_model ===============================
    #
    #
    def set_model(self, pCutModel):
        '''!
        @brief  Provide the background "classification" model to use for detection. 

        @param[in]  pCutModel   Background planar cut model instance.
        '''

        self.bgModel = pCutModel


    #================================ measure ================================
    #
    #
    def measure(self, I):
        '''!
        @brief  Apply the appearance detector to an image.

        @param[in]  I   Image to test on.
        '''
        if self.processor:
            pI = self.processor.apply(I)
        else:
            pI = I

        if self.bgModel.vectorize:
            imDat   = np.array(pI).reshape(-1,pI.shape[2]).T
            fgB     = self.bgModel.classify(imDat)
            self.Ip = np.array(fgB).reshape(pI.shape[0], pI.shape[1])
        else:
            self.Ip = self.bgModel.classify(pI)


    #========================== build_model_blackBG ==========================
    #
    @staticmethod
    def build_model_blackBG(dist, tau, isVectorized = True):
        '''!
        @brief  Build a black color-based background model. The assumption is that
                the background colors are in the black corner of the color cube
                and can be split from the foreground colors. In this case, the
                planar cut normal is known.

        The good news about this kind of model is that it does not matter whether
        the image is RGB (typical) or BGR (OpenCV type).  The corner region containing 
        black is still around (0,0,0).

        @param[in]  dist    Locate of separating boundary (should be negative). 
        @param[in]  tau     The threshold / margin to apply.
        '''

        n = np.array([1, 1, 1])/np.sqrt(3)      # Points to white color along gray line.
                                                # Dist should be negative.

        blackBG = PlanarModel.build_model(n, dist, tau, isVectorized)
        return blackBG

    #========================== build_model_whiteBG ==========================
    #
    @staticmethod
    def build_model_whiteBG(dist, tau, isVectorized = True):
        '''!
        @brief  Build a white color-based background model. This is rare but
                provided just in case.  The sign is flipped to cut off the
                more extremal white color values.

        The good news about this kind of model is that it does not matter whether
        the image is RGB (typical) or BGR (OpenCV type).  The corner region containing 
        black is still around (0,0,0).

        @param[in]  dist    Locate of separating boundary (should be positive). 
        @param[in]  tau     The threshold / margin to apply.
        '''

        n = np.array(-[1, 1, 1])/np.sqrt(3)      # Points negative to white color along gray line.
                                                 # Dist should be positive.

        whiteBG = PlanarModel.build_model(n, dist, tau, isVectorized)
        return whiteBG


    #============================= calibrate_from_data =============================
    #
    #
    @staticmethod
    def calibrate_from_data(bgI, fgI):
     
        pass
        # @todo To be coded up.

    #============================= calibrate_from_image ============================
    #
    #
    @staticmethod
    def calibrate_from_image(I, bgI):
     
        pass
        # @todo To be coded up.

#
#============================ detector.fgmodel.inCorner ============================
