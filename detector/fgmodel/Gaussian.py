#================================ fgGaussian ===============================
#
# @author   Jun Yang,
# @author   Patricio A. Vela,   pvela@gatech.edu
#
# @date     2009/07/XX      [original Matlab]
# @date     2012/07/07      [taken Matlab version]
# @date     2023/06/08      [converted to python]
#
# Version   1.0
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#================================ fgGaussian ===============================

from dataclasses import dataclass
import numpy as np
import cv2
import h5py

from detector.inImage import fgImage
from detector.Configuration import AlgConfig
import camera.utils.display as display
#import ivapy.display_cv as display


@dataclass
class SGMstate:
  """!
  @brief    Data class for storing Gaussian foreground output.
  """
  fgIm  : np.ndarray

@dataclass
class SGMdebug:
  """!
  @brief    Data class for storing Gaussian mean and variance diagonal.
  """
  mu    : np.ndarray
  sigma : np.ndarray


#
#-------------------------------------------------------------------------
#====================== Gaussian Configuration Node ======================
#-------------------------------------------------------------------------
#

class CfgSGT(AlgConfig):
  '''!
  @brief    Configuration setting specifier for Gaussian BG model.

  The most basic settings are the mean and variance vectors (mu, sigma),
  where the assumption is that the covariance matrix is diagonal only,
  and the detection threshold.  Also available are the learning rate (alpha),
  the minimum sigma permissible and the minimum area of the foreground target.
  The minimum sigma prevents small values that end up behaving like a delta
  function and are too restrictive regarding what is acceptable.

  @note     Currently not using a CfgBGModel super class. Probably best to do so.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgSGT.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #========================= get_default_settings ========================
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''

    default_dict = dict(tauSigma = 4.0, minSigma = [50.0], alpha = 0.05, \
                        init = dict( sigma = [20.0] , mu = None), \
                        minArea = 0)
    return default_dict

  #========================== builtForLearning =========================
  #
  #
  @staticmethod
  def builtForLearning():
    learnCfg = CfgSGT();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = [200.0]
    return learnCfg

  #=========================== builtForRedGlove ==========================
  #
  # minArea  = 5000    # For 1920x1200 resolution capture.
  # minArea  = 500     # For 848x480 resolution capture. [Default]
  # 
  @staticmethod
  def builtForRedGlove(minArea = 500, initModel = None):
    learnCfg = CfgSGT();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = [900.0, 100.0, 150.0]
    if (initModel is None):
      learnCfg.init.mu    = [130.0, 10.0, 50.0]
      learnCfg.init.sigma = [1200.0, 150.0, 350.0]
    else:
      learnCfg.init.mu    = initModel[0]
      learnCfg.init.sigma = initModel[1]

    learnCfg.minArea  = 800     
    return learnCfg

  #========================== builtForDepth435 =========================
  #
  #
  @staticmethod
  def builtForDepth435():
    depth_dict = dict(tauSigma = 1.0, minSigma = 0.0004, alpha = 0.05, \
                        init = dict( sigma = [0.0002] , mu = None)  )
    learnCfg = CfgSGT(depth_dict);
    return learnCfg

#
#---------------------------------------------------------------------------
#================================ fgGaussian ===============================
#---------------------------------------------------------------------------
#

class fgGaussian(fgImage):
  """!
  @ingroup  Detector
  @brief    Single Gaussian target/foreground model with diagonal covariance.

  A similar implementation exists in ``targetSG``, based on a different
  operating paradigm that decorrelates or whitens the input image data, which
  is effectively a linear transformation of the image data.  While the
  implementation is most likely better than this one, simplicity has its own
  value.

  No doubt this implementation exists in some form within the OpenCV or BGS
  libraries, but getting a clean, simple interface from these libraries is
  actually not as easy as implementing from existing Matlab code.  Plus, it
  permits some customization that the library implementations may not have.

  For how to configure the input, please see CfgSGT.  Likewise for model
  overriding, please see SGMdebug.

  @note  A note on the improcessor.  If the basic version is used, then it
    performs pre-processing.  If a triple version is used, then the
    mid-processor will perform operations on the detected part rather
    than the default operations.  The mid-processor can be used to test
    out different options for cleaning up the binary data.

  @todo Eventually should be translated to OpenCV style code by repurposing
        their code to get CUDA and OpenCL implementations for speed purposes.
        Even their C code is fairly zippy relative to python.

  @todo Redo so that inherits from appearance or some kind of fgmodel class.
  """

  #======================== fgGaussian/__init__ ========================
  #
  #
  def __init__(self, bgCfg = None, processor = None, fgMod = None):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    @param[in]  bgMod   The model or parameters for the detector.
    @param[in]  bgCfg   The model or parameters for the detector.
    '''
    super(fgGaussian, self).__init__(processor)

    # First, set the configuration member field.
    if bgCfg is None:
      self.config = CfgSGT()
    else:
      self.config = bgCfg

    # Set all runtime member variables and working memory.
    self.imsize = None

    # Foreground model.  
    # Apply YAML/config defaults if available.
    # Providing a foreground model in fgMode overrides these values.
    if self.config.init.mu is not None:
      self.mu   = np.array(self.config.init.mu)
    else:
      self.mu   = None

    if self.config.init.sigma is not None:
      self.sigma  = np.array(self.config.init.sigma)
    else:
      self.sigma = None

    # Last measurement.
    self.measI = None
    self.measM = None

    # Working memory.
    self.errI  = None
    self.sqeI  = None
    self.nrmE  = None
    self.maxE  = None
    self.fgI   = None

    # Check for image processor routine.
    self.improcessor = processor

    # If foreground model passed in, set it.
    if fgMod is not None:
      self.mu    = fgMod.mu
      self.sigma = fgMod.sigma


  #============================= _setsize_ =============================
  #
  def _setsize_(self, imsize):

    if (np.size(imsize) == 2):
      self.imsize = np.append(imsize, [1])
    else:
      self.imsize = imsize;

    self._preallocate_()

  #=========================== _preallocate_ ===========================
  #
  def _preallocate_(self):
    '''!
    @brief  Image size known, so instantiate memory for working variables.
    '''

    if self.imsize is None:
      return
    
    bigShape = ( np.prod(self.imsize[0:2]), self.imsize[2] )
    if (self.imsize[2] > 1):                    
      linShape = ( np.prod(self.imsize[0:2]) )
    else:
      linShape = ( np.prod(self.imsize[0:2]), 1 )

    # @todo Figure out the deal with if statement above.  My python-fu is limited.
    #       Not sure how to deal with matrices of same size but somehow not
    #       broadcastable.  Matlab wouldn't be triggering this kind of error.

    self.measI = np.zeros( bigShape )
    self.errI  = np.zeros( bigShape )
    self.sqeI  = np.zeros( bigShape )
    self.nrmE  = np.zeros( bigShape ) 
    self.maxE  = np.zeros( linShape ) 
    self.fgI   = np.zeros( linShape , dtype=bool) 

  #============================== predict ==============================
  #
  def predict(self):
    '''!
    @brief  Predictive model of measurement.

    In standard schemes, the expectation is that the background model
    is static (a constant state model).  Thus, the default prediction is no
    update.
    '''

    pass

  #============================== measure ==============================
  #
  # @todo   Need to see NumExpr library for faster numerical expression evaluation.
  #         See [here](https://github.com/pydata/numexpr).
  #
  # Currently using numpy routines for in-place computation so that memory
  # allocation can be avoided.
  #
  def measure(self, I, M = None):
    '''!
    @brief    Takes image and generates the detection result.
  
    @param[in]    I   Image to process.
    '''
    if self.improcessor is not None: 
      I = self.improcessor.pre(I)
    
    if self.imsize is None:
        self._setsize_(np.array(np.shape(I)))

    self.measI = np.array(I, dtype=float, copy=True)
    self.measI = np.reshape(self.measI, 
                            np.append(np.prod(self.imsize[0:2]), self.imsize[2]) )


    #if self.mu is None:
      #self.mu = self.measI.copy()
      
    # sqeI = (mu - measI).^2 / sigma  (in Matlab paraphrasing).
    # Apply operations wih broadcasting to avoid memory re-allocation.
    # Store outcomes since they get used in the adapt routine.
    np.subtract( self.measI, self.mu, out=self.errI )
    np.square  ( self.errI , out=self.sqeI )
    np.divide  ( self.sqeI , self.sigma, out=self.nrmE )

    #DEBUG
    #print(np.shape(self.measI))
    #print(np.shape(self.sqeI))
    #print(np.shape(self.nrmE))
    #print(np.shape(self.sigma))
    #print('------')

    # Find max error across dimensions if there are more than 1,
    # as would occur for an RGB image.
    #
    if (self.imsize[2] > 1):
      #DEBUG
      #print(np.shape(self.nrmE))
      #print(np.shape(self.maxE))
      #print(np.shape(np.amax(self.nrmE, axis=1)))
      #print('xxxxxxxxxxxxx')
      np.amax( self.nrmE, axis=1, out=self.maxE )
    else:
      np.copyto(self.maxE, self.nrmE )

    np.less( self.maxE, self.config.tauSigma, out=self.fgI )
    if M is not None:
      np.logical_and( self.fgI, np.ndarray.flatten(M) ,  out=self.fgI )
  
    if self.improcessor is not None:
      self.bgI = bgp.improcessor.post(self.bgI)
  

  #============================== correct ==============================
  #
  def correct(self):
    '''!
    @brief  Generate a correction to the model.

    In standard schemes, there are no corrections to the estimates.  The
    classification result is presumed to be correct.  Corrections would
    imply some sort of temporal regularization.  Spatial regularization
    is usually done through image-based mid-processing.
    '''

    pass


  #=============================== adapt ===============================
  #
  def adapt(self):
    '''!
    @brief  Update the Gaussian model based on recent measurement.

    In this case, the mean and the variance are updated.  Depending on
    the run-time options, all means/variances will be updated or only
    those classified as background.  The latter avoids adapting to
    foreground elements while still permitting slow change of the 
    background model.  

    Usually, during the model estimation phase (assuming an empty scene
    with background elements only) adaptation of all pixels should occur.
    During deployment, if adaptation is to be performed, then it is usually
    best to not apply model updating to foreground elements, which are
    interpreted as fast change elements of the scene.  
    '''

    if self.config.alpha == 0:
        return

    # MOST LIKELY NOT NECESSARY FOR FG MODEL. DELETE SHORTLY.
    # DELETED
    #if not self.config.adaptall:                # Get foreground pixels.
    #  fginds   = np.nonzero(~self.bgI);                          
    #  oldmu    = self.mu[fginds,:];             # Save current values. 
    #  oldsigma = self.sigma[fginds,:];

    # Snag any target/foreground pixels and perform update.
    # Update mean and variance. @todo NEED TO FIX.
    # mu    = (1 - alpha) mu + alpha * newmu     = mu     + alpha*(newmu - mu)
    # sigma = (1 - alpha) sigma + alpha * newsig = sigma  + alpha*(newsign - sigma)
    #
    #DEBUG
    #print(np.shape(self.measI), ' and ', np.shape(self.fgI))

    if (self.imsize[2] > 1):
      newVals = self.measI[self.fgI,:]

      if (np.size(newVals) > 0):
        # @todo Establish is data is row-wise or column-wise, then document here.
        #       Should always note expected data organization to prevent problems.
        newMu   = np.mean(newVals, 0)
        newSig  = np.var(newVals, 0)

        self.mu    = self.mu    + self.config.alpha*(newMu - self.mu)
        self.sigma = self.sigma + self.config.alpha*(newSig - self.sigma)

        # Impose min sigma constraint.
        np.maximum(self.sigma, self.config.minSigma, out=self.sigma)

    else:
      newVals = self.measI[self.fgI]

      if (np.size(newVals) > 0):
        newMu   = np.mean(newVals)
        newSig  = np.var(newVals)
     
        self.mu    = self.mu    + self.config.alpha*(newMu - self.mu)
        self.sigma = self.sigma + self.config.alpha*(newSig - self.sigma)

        # Impose min sigma constraint.
        np.maximum(self.sigma, self.config.minSigma, out=self.sigma)
  
    # MOST LIKELY NOT NECESSARY FOR FG MODEL. DELETE SHORTLY.
    #if not self.config.adaptall:                # Revert foreground values.
    #  self.mu[fginds,:]    = oldmu      
    #  self.sigma[fginds,:] = oldsigma

  #=============================== detect ==============================
  #
  def detect(self, I, M = None):
    '''!
    @brief  Given a new measurement, apply the detection pipeline.

    This only goes so far as to process the image and generate the
    detection, with correction.  There is no adaptation. It should
    be run separately if desired.
    '''

    self.predict()
    self.measure(I, M)
    self.correct()

  #============================== process ==============================
  #
  def process(self, I):
    '''!
    @brief  Given a new measurement, apply entire FG modeling pipeline.

    @param[in]  I   New image measurement.
    '''
  
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

  #============================= emptyState ============================
  #
  def emptyState(self):

    eState = SGMstate
    return eState

  #============================= emptyDebug ============================
  #
  def emptyDebug(self):

    eDebug = SGMdebug
    return eDebug

  #============================== getState =============================
  #
  def getState(self):

    cState = SGMstate(fgIm = self.fgI.reshape(self.imsize[0:2]))
    return cState

  #============================== getDebug =============================
  #
  def getDebug(self):

    cDebug = SGMdebug(mu = self.mu.reshape(self.imsize), 
                      sigma = self.sigma.reshape(self.imsize) ) 
    return cDebug

  #=========================== displayState ============================
  #
  def displayState(self):
    pass

  #=========================== displayDebug ============================
  #
  def displayState(self):
    pass

  #================================ set ================================
  #
  def set(self):
    pass

  #================================ get ================================
  #
  def get(self):
    pass

  #================================ info ===============================
  #
  def info(self):
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

  #========================== estimateFromData =========================
  #
  def estimateFromData(self, theData):
    """!
    @brief  Use given data to estimate Gaussia foreground model.

    @param[in]  theData     Column-vectors of target data.

    @todo   Establish if should be row or column.
    """
    
    self.mu    = np.mean(theData, axis=1)
    self.sigma = np.var(theData, axis=1)

    np.maximum(self.sigma, self.config.minSigma, out=self.sigma)

  #=========================== updateFromData ==========================
  #
  def updateFromData(self, theData, alpha = None):
    """!
    @brief  Use given data to update Gaussia foreground model based on learning rate.

    @param[in]  theData     Column-vectors of target data.
    @param[in]  alpha       [None] Learning rate override: set for value other than internal one.

    @todo   Establish if should be row or column.
    """
    
    if (theData is None) or (theData.size == 0):
      return

    newMu  = np.mean(theData, axis=1)
    newSig = np.var(theData, axis=1)

    if alpha is None:
      alpha = self.config.alpha

    if self.mu is None:
      self.mu    = newMu
    else:
      self.mu    = self.mu    + alpha*(newMu - self.mu)


    if self.sigma is None:
      self.sigma = newSig
    else:
      self.sigma = self.sigma + alpha*(newSig - self.sigma)
      
    # Impose min sigma constraint.
    np.maximum(self.sigma, self.config.minSigma, out=self.sigma)


  #======================== estimateFromMaskRGB ========================
  #
  def estimateFromMaskRGB(self, theMask, theImage):
    """!
    @brief  Extract mask region pixel data from image for Gaussia model
            estimation.

    @param[in]  theMask     Regions of interest w/binary true values.
    @param[in]  theImage    Source image to get model data from.
    """

    masize   = np.shape(theMask)
    imsize   = np.shape(theImage)

    if any(masize != imsize[0:2]):
      return
      # @todo   What to do in case of bad args?

    # Get mask elements from image. Requires reshaping image to have vectorized
    # data, getting the indices from the mask, then computing the model
    # statistics. First prep for collecting the vectorized data.
    vecImage = np.array(theImage, dtype=float, copy=True)
    vecImage = np.reshape( vecImage, np.append(np.prod(imsize[0:2]), imsize[2]) )

    vecMask  = theMask.flatten()

    # Get vectorized data and send to estimation routine.
    theData = np.transpose(vecImage[vecMask,:])
    self.estimateFromData(theData)

  #========================= updateFromMaskRGB =========================
  #
  def updateFromMaskRGB(self, theMask, theImage, alpha = None):
    """!
    @brief  Extract mask region pixel data from image for Gaussia model
            update/adaptation.

    @param[in]  theMask     Regions of interest w/binary true values.
    @param[in]  theImage    Source image to get model data from.
    @param[in]  alpha       [None] Learning rate override: set for value other than internal one.
    """

    masize   = np.shape(theMask)
    imsize   = np.shape(theImage)

    if (masize != imsize[0:2]):
      return
      # @todo   What to do in case of bad args?

    # Get mask elements from image. Requires reshaping image to have vectorized
    # data, getting the indices from the mask, then computing the model
    # statistics. First prep for collecting the vectorized data.
    vecImage = np.array(theImage, dtype=float, copy=True)
    vecImage = np.reshape( vecImage, np.append(np.prod(imsize[0:2]), imsize[2]) )

    vecMask  = theMask.flatten()

    # Get vectorized data and send to estimation routine.
    theData = np.transpose(vecImage[vecMask,:])
    self.updateFromData(theData)

  #======================== refineFromStreamRGB ========================
  #
  #
  def refineFromStreamRGB(self, theStream, incVis = False):
    """!
    @brief  Given an RGB stream, run the estimation process with 
            adaptation on to improve color model.
    
    @param[in]  theStream   RGB stream.
    @param[in]  incVis      Include visualization of refinement processing [False]
    """

    print('STEPS to Refine the Gaussian model.')
    print('\t [1] Hit any key to continue once scene is prepped.')
    print('\t [2] Wait a little. Hit "q" to stop adaptation process. Should be short.')
    input();
  
    while(True):
      rgb, success = theStream.capture()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()
  
      self.process(rgb)
  
      if (incVis):
        fgS = self.getState()
        display.rgb_binary_cv(rgb, fgS.fgIm, 0.25, "Output")
  
      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
          break
  
    if (incVis):
      display.close_cv("Output")

  #======================== refineFromRGBDStream =======================
  #
  # @brief  Given an RGBD stream, run the estimation process with 
  #         adaptation on to improve color model.
  #
  def refineFromRGBDStream(self, theStream, incVis = False):

    print('STEPS to Refine the Gaussian model.')
    print('\t [1] Hit any key to continue once scene is prepped.')
    print('\t [2] Wait a little. Hit "q" to stop adaptation process. Should be short.')
    input();
  
    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()
  
      self.process(rgb)
  
      if (incVis):
        fgS = self.getState()
        display.rgb_binary_cv(rgb, fgS.fgIm, 0.25, "Output")
  
      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
          break
  
    if (incVis):
      display.close_cv("Output")

  #========================== testOnRGBDStream =========================
  #
  # @brief  Given an RGBD stream, run the detection process to see how
  #         well detection works (or not).
  #
  def testOnRGBDStream(self, theStream, incVis = True):

    print('STEP: \n\t Test out current foreground model.')
    print('\t Hit "q" to stop testing process.')
  
    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()
  
      self.detect(rgb)
  
      if (incVis):
        fgS = self.getState()
        display.rgb_binary_cv(rgb, fgS.fgIm, 0.25, "Test")
  
      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
          break
  
    print('\t Done.')
    if (incVis):
      display.close_cv("Test")


  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    self.config.init.mu    = self.mu.tolist()
    self.config.init.sigma = self.sigma.tolist()

    fPtr.create_dataset("ForegroundGaussian", data=self.config.dump())

  #============================ saveConfig =============================
  #
  #
  def saveConfig(self, outFile): 
    '''!
    @brief  Save current instance to a configuration file.
    '''

    self.config.init.mu    = self.mu.tolist()
    self.config.init.sigma = self.sigma.tolist()

    with open(outFile,'w') as file:
      file.write(self.config.dump())
      file.close()


  #
  #-----------------------------------------------------------------------
  #======================= Static Member Functions =======================
  #-----------------------------------------------------------------------
  #

  #============================ loadFromYAML ===========================
  #
  #
  @staticmethod
  def loadFromYAML(fileName):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''

    theSetup = CfgSGT()
    theSetup.merge_from_file(fileName)
    fgDetector = fgGaussian(theSetup)
    return fgDetector

  #============================ buildFromCfg ===========================
  #
  #
  @staticmethod
  def buildFromCfg(theConfig, processor = None, fgMod = None):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''

    fgDetector = fgGaussian(theConfig, processor, fgMod)
    return fgDetector

  #================================ load ===============================
  #
  @staticmethod
  def load(fileName):
    """!
    @brief  Load Gaussian instance specification from HDF5 file.
    """
    fptr = h5py.File(fileName,"r")
    keyList = list(fptr.keys())
    theModel = fgGaussian.loadFrom(fptr)
    fptr.close()
    return theModel


  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fPtr):
    """!
    @brief  Specialzed load function to parse HDF5 data from file pointer.
    """
    keyList = list(fPtr.keys())

    # @todo     May not be loading from actual model but from config. Not good.
    if ("ForegroundGaussian" in keyList):
      cfgPtr = fPtr.get("ForegroundGaussian")
      cfgStr = cfgPtr[()].decode()

      theConfig = CfgSGT.load_cfg(cfgStr)
    else:
      print("No foreground Gaussian model; Returning None.")
      theConfig = None

    theModel = fgGaussian(theConfig)
    return theModel

#
#================================ fgGaussian ===============================
