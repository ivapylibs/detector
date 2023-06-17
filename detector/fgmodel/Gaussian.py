#================================ Gaussian ===============================
"""
  @class Gaussian

  @brief    Implements a single Gaussian target/foreground model.

  A similar implementation exists in ``targetSG``, based on a different
  operating paradigm that ddecorrelates the input image data.  While
  the implementation is most likely better than this one,

  no doubt going to be 
  No doubt this implementation exists in some form within the OpenCV or
  BGS libraries, but getting a clean, simple interface from these libraries
  is actually not as easy as implementing from existing Matlab code.
  Plus, it permits some customization that the library implementations
  may not have.

  Inputs:
    mu          - the means of the Gaussian model.
    sigma       - the variance of the Gaussian model.
    weights     - the weights of the Gaussian models.
    parms       - [optional] configuration instance with parameters specified.

  Fields of the parms structure:
    sigma       - Initial variance to use if sigma is empty.
    thresh      - Threshold for determining foreground.
    alpha       - Update rate for mean and variance.

    A note on the improcessor.  If the basic version is used, then it
    performs pre-processing.  If a triple version is used, then the
    mid-processor will perform operations on the detected part rather
    than the default operations.  The mid-processor can be used to test
    out different options for cleaning up the binary data.
"""
#================================ Gaussian ===============================
#
# @file     Gaussian.py
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
#================================ Gaussian ===============================

from yacs.config import CfgNode
from dataclasses import dataclass
import numpy as np

from detector.inImage import inImage

@dataclass
class SGMstate:
  fgIm  : np.ndarray

@dataclass
class SGMdebug:
  mu    : np.ndarray
  sigma : np.ndarray


class CfgSGT(CfgNode):
  '''!
  @brief  Configuration setting specifier for Gaussian BG model.

  @note   Currently not using a CfgBGModel super class. Probably best to do so.
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

    default_dict = dict(tauSigma = 4.0, minSigma = 50.0, alpha = 0.05, \
                        adaptall = False,
                        init = dict( sigma = 20.0 , imsize = None)  )
    return default_dict

  #========================== builtForLearning =========================
  #
  #
  @staticmethod
  def builtForLearning():
    learnCfg = CfgSGT();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = 200.0
    return learnCfg

  #=========================== builtForRedGlove ==========================
  #
  #
  @staticmethod
  def builtForRedGlove():
    learnCfg = CfgSGT();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = [900, 100, 150]
    return learnCfg

  #========================== builtForDepth435 =========================
  #
  #
  @staticmethod
  def builtForDepth435():
    depth_dict = dict(tauSigma = 1.0, minSigma = 0.0004, alpha = 0.05, \
                        adaptall = False,
                        init = dict( sigma = 0.0002 , imsize = None)  )
    learnCfg = CfgSGT(depth_dict);
    return learnCfg

#================================ Gaussian ===============================

class Gaussian(inImage):

  #========================= Gaussian/__init__ =========================
  #
  #
  def __init__(self, bgCfg = None, processor = None, fgMod = None):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    @param[in]  bgMod   The model or parameters for the detector.
    @param[in]  bgCfg   The model or parameters for the detector.
    '''
    super(Gaussian, self).__init__(processor)

    # First, set the configuration member field.
    if bgCfg is None:
      self.config = CfgSGT()
    else:
      self.config = bgCfg

    # Set all runtime member variables and working memory.
    self.imsize = self.config.init.imsize

    # Foreground model.
    self.mu     = None
    self.sigma  = None

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
  # @todo   Need to se NumExpr library for faster numerical expression evaluation.
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
      newMu   = np.mean(newVals, 0)
      newSig  = np.var(newVals, 0)
    else:
      newVals = self.measI[self.fgI]
      newMu   = np.mean(newVals)
      newSig  = np.var(newVals)
     
    #DEBUG
    #print(np.size(newVals))
    #print(np.shape(newVals))
    #print(np.shape(self.mu))
    if (np.size(newVals) > 0):
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
                      #errIm = self.maxE.reshape(self.imsize[0:2]))
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

  #================================ save ===============================
  #
  def save(self, fileName):    # Save given file.
    pass

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    pass

  #============================== saveCfg ============================== 
  #
  def saveCfG(self, outFile): # Save to YAML file.
    pas


  #================================ load ===============================
  #
  @staticmethod
  def load(fileName):
    pass

  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fileName):
    pass


#
#================================ Gaussian ===============================
