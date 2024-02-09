#=============================== bgmodelGMM ==============================
#
#
# @author   Gbolabo Ogunmakin, 		gogunmakin3@gatech.edu
# @author	Patricio A. Vela,		pvela@gatech.edu
# @author   Yiye Chen (py),         yychen2019@gatech.edu
# 
# @date   2011/01/31 		(original: bg_model.m)
# @date   2021/07/22      (python) 
# 
#! NOTE:
#!    set tabstop = 4, indent = 2.
#!    90 columns text.
#=============================== bgmodelGMM ==============================

from dataclasses import dataclass
import numpy as np
import cv2
import h5py

from detector.inImage import bgImage
from detector.Configuration import AlgConfig
import ivapy.display_cv as display

@dataclass
class GMMstate:
  bgIm  : np.ndarray
  fgIm  : np.ndarray

@dataclass
class GMMdebug:
  mu    : np.ndarray
  sigma : np.ndarray
  errIm : np.ndarray

@dataclass
class Params_cv:
    """!
    The parameters for the bgmodelGMM_cv

    For mored detail on the parameters, check OpenCV documentation.
    @param  history         @< Number of frames to use for model estimation
    @param  NMixtures       @< Number of mixtures
    @param  varThreshold    @< Variance threshold for detection.
    @param  detectShadow    @< Boolean: detect shadows or not.
    @param  adapt_rate      @< Adaptation rate.

    @note   Deprecated. Should not be used. Will be removed at some point.
            Instead use CfgGMM_cv for parameters/configuration.
    """
    history:    int = 300
    NMixtures:  int = 5
    varThreshold:       float = 50.
    detectShadows:      bool = True
    ShadowThreshold:    float = 0.5
    adapt_rate:         float = -1      # will automatically choose rate



class CfgGMM_cv(AlgConfig):
  '''!
  @ingroup  Detector_BGModel
  @brief    Configuration setting specifier for Gaussian BG model.

  @note     Currently not using a CfgBGModel super class. Should there be one?
  @note     What about config from bgImage class? Doesn't exist as of [24/02/08]
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgGMM_cv.get_default_settings()

    super(CfgGMM_cv,self).__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #========================= get_default_settings ========================
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    Using default settings that were found to work well for IVALab setups.
    Something has to be used as default.  Actual deployment should be based
    on saved calibration of deployment scene.  In our case, detecting shadows
    is not particularly helpful, hence the default of False for that case.

    Due to OpenCV implementation, no adaptation involved turning off the
    update by setting the learning rate to 0 (i.e., alpha = 0).

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''
    default_dict = dict(history  = 300, NMixtures = 5, 
                        tauSigma = 50.0, minSigma = 25.0, maxSigma = 10000.0,
                        alpha = -1,
                        init = dict( sigma = 40.0, imsize = [] ),
                        detectShadow = False, tauShadow = 0.5 ) 

    return default_dict


  #========================== builtForLearning =========================
  #
  #
  @staticmethod
  def builtForLearning():
    learnCfg = CfgGMM_cv();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = [4.0]
    return learnCfg


#================================ bgmodelGMM ===============================
class bgmodelGMM(bgImage):
    """
    @ingroup    Detector_BGModel
    @brief      Gaussian mixture background model.

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

    @note
        A note on the improcessor.  If the basic version is used, then
		it performs pre-processing. If a triple version is used, then
		the mid-processor will perform operations on the detected part
		rather than the default operations.  The mid-processor can be used
		to test out different options for cleaning up the binary data.

    @note Not implemented at all. Translation of ivaMatlib/bgmodelGMM
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

#============================== bgmodelGMM_cv ==============================
#

class bgmodelGMM_cv(bgImage):
    """
    @brief  GMM Background Substraction method MOG2 from OpenCV library.

    Wrapper for the OpenCV Gaussian mixture model foreground detection
    implementation.  This model involves two parts, one is an estimator, and
    the other is a change detector.  The estimator is a static prediction
    observer on the mean and variance with fixed udpate gains.  The change
    detector is threshold-based.  

    The detection algorithm will first use the GMM to detect a potential
    foreground mask, each pixel of which will be checked for the color
    distort and the intensity decay. 

    The shadow detection method is from [paper](http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf)
    based on [OpenCV code](https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp#L477-L520).
    """

    #=============================== __init__ ==============================
    #
    def __init__(self, theConfig, theProcessor = None):

        if (theConfig is None):
          theConfig = CfgGMM_cv

        super().__init__()
        self.config = theConfig

        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.tauSigma,
            detectShadows=self.config.detectShadow 
        )
        self.set("ShadowThreshold", self.config.tauShadow)
        self.set("NMixtures", self.config.NMixtures)

        # parameters for apply
        self.adapt_rate = self.config.alpha 

        # for storing the result
        self.detResult      = None
        self.shadow_mask    = None
        self.fg_mask        = None

        self.bgSubtractor.setVarMin(4.0) #self.config.minSigma) 
        self.bgSubtractor.setVarMax(self.config.maxSigma)
        self.bgSubtractor.setVarInit(self.config.init.sigma)


    #=============================== measure ===============================
    #
    def measure(self, I):
        """!
        @brief  Does nothing. Incompatible with OpenCV interface.
        """
        pass

    #=============================== correct ===============================
    #
    def correct(self, fg):
        """!
        @brief  Does nothing. Incompatible with OpenCV interface.
        """
        pass
    
    #================================ adapt ================================
    #
    def adapt(self):
        """!
        @brief  Does nothing. Incompatible with OpenCV interface.
        """
        pass

    #================================ detect ===============================
    #
    def detect(self, img):
        """
        @brief  Apply detection w/correction but do not adapt background model.

        Overrides the learning rate (alpha) in the configuration by simply
        ignoring it and passing on no learning to the OpenCV implementation.
        """
        self.detResult = self.bgSubtractor.apply(img, learningRate=0)
        self.fg_mask   = (self.detResult == 255)

        if self.config.detectShadow:
          self.shadow_mask = (self.detResult == 127)

    #=============================== process ===============================
    #
    def process(self, img):
        """!
        @brief  Process image through OpenCV background subtractor.

        This is only way to invoke entire process.  There is no ability to
        call the intermediate processes (measure, correct, adapt, etc.).
        Do not call the other member functions as they do nothing.
        """

        self.detResult = self.bgSubtractor.apply(img, learningRate=self.config.alpha) 
        self.fg_mask   = (self.detResult == 255)

        if self.config.detectShadow:
          self.shadow_mask = (self.detResult == 127)

    #=================================== set =================================
    #
    def set(self, fname, fval):
        """!
        Check the [documentation](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#acdb85152b349d70561fecc4554ad76e6)
        for what can be set.

        @param[in]  fname   Name of parameter to set. Invoke set+fname for opencv MOG2
        @param[in]  fval    Value to set

        Example:

        det = bgmodelGMM_cv()
        det.set("History", 200)   # will invoke setHistory(200) function from the link
        """
        eval( "self.bgSubtractor.set" + fname + "(" + str(fval) + ")" )

    
    #=================================== get =================================
    #
    def get(self, fname):
        """
        Check the [documentation](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#acdb85152b349d70561fecc4554ad76e6)
        what parameters to get.

        @param[in]  fname   Name of parameter to get. Invoke get+fname for opencv MOG2

        example:

        det = bgmodelGMM_cv()
        det.get("History")          # will invode getHistory() function from the link
        """
        fval = eval( "self.bgSubtractor.get" + fname + "()" )
        return fval

    #=============================== getState ==============================
    #
    def getState(self):
        """!
        @brief    Get latest detection result stored in memory.
        """
        cState = GMMstate(bgIm = self.detResult, fgIm = self.fg_mask)
        return cState

    #=============================== getDebug ==============================
    #
    def getDebug(self):
        """!
        @brief    Get latest detection result stored in memory.
        """
        cState = GMMdebug(mu = self.getBackgroundImage(), sigma = None, errIm = None)
        return cState

#    def getDetectResult(self):
#        """
#        Get the detection result, including the foreground and the shadow
#        The foreground pixels' value will be 255, wherease the shadow pixels' will be 127. So the shadows will look like shadow(darker)
#        """
#        return self.detResult
    
#    def getForeground(self):
#        """
#        Get the current foreground estimate
#        """
#        return self.fg_mask
#    
#    def getBackground(self):
#        """
#        Get the current background mask
#        """
#        return ~self.fg_mask
#
#    def getShadow(self):
#        """
#        Get the detected shadow mask
#        """
#        return self.shadow_mask
#
#    def getProbs(self):
#        """
#        Get the current probability and the generating model
#        """
#        pass
    
    #=========================== getBackgroundImg ==========================
    #
    def getBackgroundImage(self):
        """!
        @brief  Get the background image in RGB, based on the background GMM model.

        The image will be the [weighted mean of the Gaussians'
        means](https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp#L887-L927)

        @return     Background RGB image; bgImg (H, W, 3).
        """
        bgmImage = self.bgSubtractor.getBackgroundImage()
        print(bgmImage)
        return self.bgSubtractor.getBackgroundImage()[:,:,::-1]

    #=================== buildAndCalibrateFromConfigRGBD ===================
    #
    # @brief  build and calibrate onWorkspace model from an initial config 
    #         and a camera class streaming camera. Return instantiated and 
    #         calibrated model.
    #         
    # The stream is presumed to be a depth + color stream as obtained from
    # a Realsense camera.  Code is not as generic as could be.
    #
    # @todo   Modify to be a bit more generic. [24/02/08 : What does this mean?]
    # @todo What about the image processor?? Dump into the config??
    #
    #
    @staticmethod
    def buildAndCalibrateFromConfigRGBD(theConfig, theStream, incVis = False):
  
        print('\n STEPS to calibrate onWorkspace.')
        print('\t [1] Make sure workspace is empty.')
        print('\t [2] Hit enter to continue once scene is prepped.')
        print('\t [3] Hit "q" to stop adaptation process. Should be short.')
        input()

        bgModel = bgmodelGMM_cv( theConfig )
     
        while(True):
          rgb, dep, success = theStream.get_frames()
          if not success:
            print("Cannot get the camera signals. Exiting...")
            exit()
    
          bgModel.process(rgb)
    
          if (incVis):
            bgS = bgModel.getState()
            bgD = bgModel.getDebug()
    
            bgIm  = cv2.cvtColor(bgS.bgIm, cv2.COLOR_GRAY2BGR)
            bgMod = bgD.mu.astype(np.uint8)
            #bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            display.rgb(bgIm, ratio=0.25, window_name="Detection")
            display.rgb(bgMod, ratio=0.25, window_name="BG Model")
    
          opKey = cv2.waitKey(1)
          if opKey == ord('q'):
            break
       
        display.close("Detection")
        display.close("BG Model")
        return bgModel

    #==================== buildAndCalibrateFromConfigRGB ===================
    #
    # @brief  build and calibrate onWorkspace model from an initial config 
    #         and a camera class streaming camera. Return instantiated and 
    #         calibrated model.
    #         
    # The stream is presumed to be a depth + color stream as obtained from
    # a Realsense camera.  Code is not as generic as could be.
    #
    # @todo What about the image processor?? Dump into the config??
    #
    @staticmethod
    def buildAndCalibrateFromConfigRGB(theConfig, theStream, incVis = False):
  
        print('\n STEPS to calibrate onWorkspace.')
        print('\t [1] Make sure workspace is empty.')
        print('\t [2] Hit enter to continue once scene is prepped.')
        print('\t [3] Hit "q" to stop adaptation process. Should be short.')
        input()

        bgModel = bgmodelGMM_cv( theConfig )
     
        while(True):
          rgb, success = theStream.get_frame()
          if not success:
            print("Cannot get the camera signals. Exiting...")
            exit()
    
          bgModel.process(rgb)
    
          if (incVis):
            bgS = bgModel.getState()
            bgD = bgModel.getDebug()
    
            bgIm  = cv2.cvtColor(bgS.bgIm, cv2.COLOR_GRAY2BGR)
            bgMod = bgD.mu.astype(np.uint8)
            #bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            display.rgb(bgIm,   ratio=0.25, window_name="Detection")
            display.rgb(bgD.mu, ratio=0.25, window_name="BG Model")
    
          opKey = cv2.waitKey(1)
          if opKey == ord('q'):
            break
       
        display.close_cv("Detection")
        display.close_cv("BG Model")
        return bgModel
    
    
    #============================== saveModel ==============================
    #
    def saveModel(self, fName):
        """!
        @brief  Save current MOG2 model parameters to file.
    
        Uses the native/OpenCV implementation, thus it requires a filename
        and not a filepointer.  Will save to the specified file.
    
        @note   This code is untested.  Assumes that OpenCV python API mirrors
                the actual C++ implementation.
        """
        print(fName)
        print(type(self))
        print(self.bgSubtractor)
        fs = cv2.FileStorage(fName, 1)
        self.bgSubtractor.write(fs)
  
    #================================ save ===============================
    #
    def save(self, fileName, datFileName = None):    # Save given file.
        """!
        @brief  Outer method for saving to a file given as a string.

        Opens file, preps for saving, invokes save routine, then closes.
        Usually not overloaded.  Overload the saveTo member function.
        """
        warning("Saving does not save the raw binary data for the model.")

        if (datFileName is None):
          # @todo   Need to redo.  Make better. OK for now to confirm save/load.
          datFileName = fileName.replace("hdf5", "dat")
 
        fptr = h5py.File(fileName,"w")
        self.saveTo(fptr, datFileName);
        fptr.close()

    #================================ saveTo ===============================
    #
    def saveTo(self, fPtr, fName):
        """!
        @brief  Save current MOG2 model to file.
    
        Uses the native/OpenCV implementation, thus it requires a filename
        and not a filepointer.  Will save to the specified file. Also stored
        in the HDF5 file for loading later on.
    
        @param[in]  fPtr    HDF5 file pointer (opened and ready).
        @param[in]  fName   Filename to save GMM model to.
    
        @note   This code is untested.  Assumes that OpenCV python API mirrors
                the actual C++ implementation.
        """
        wsds = fPtr.create_group("bgmodel.GMM_cv")
    
        self.saveModel(fName)
        wsds.create_dataset("sourcefile", data=fName)
    
        configStr = self.config.dump()
        wsds.create_dataset("configuration", data=configStr)
  
  
    #============================== loadModel ==============================
    #
    def loadModel(self, fName):
        """!
        @brief  Load MOG2 model to file and overwrite current model.
    
        Uses the native/OpenCV implementation, thus it requires a filename
        and not a filepointer.  Will load from the specified file.
    
        @note   This code is untested.  Assumes that OpenCV python API mirrors
                the actual C++ implementation.
        """
        fs = cv2.FileStorage(fName, 0)
        fn = fs.root()
        self.bgSubtractor.read(fn)
  
    #================================ load ===============================
    #
    @staticmethod
    def load(fileName):
        """!
        @brief  Load GMM instance from saved file.
    
        @param[in]  fileName    Source file name.
    
        @note   This code is untested.
        """
    
        warning("Loading does not load the raw binary data for the model.")
        fptr = h5py.File(fileName,"r")
        theModel = bgmodelGMM_cv.loadFrom(fptr)
        return theModel
    
  
    #============================== loadFrom =============================
    #
    @staticmethod
    def loadFrom(fPtr):
        """!
        @brief  Load GMM instance from details saved in HDF5 file.
    
        @param[in]  fileName    Source file name.
    
        @note   This code is untested.
        """
        gptr = fPtr.get("bgmodel.GMM_cv")
        
        cfgPtr     = gptr.get("configuration")
        configStr  = cfgPtr[()].decode()
    
        srcPtr     = gptr.get("sourcefile")
        srcfileStr = srcPtr[()].decode()
    
        fPtr.close()
    
        theConfig = CfgGMM_cv.load_cfg(configStr)
    
        theModel = bgmodelGMM_cv(theConfig, None)
        theModel.loadModel(srcfileStr)
    
        return theModel
