#================================ Gaussian ===============================
"""
  @class Gaussian

  @brief    Implements a single Gaussian background model.

  No doubt this implementation exists in some form within the OpenCV or
  BGS libraries, but getting a clean, simple interface from these libraries
  is actually not as easy as implementing from existing Matlab code.
  Plus, it permits some customization that the library implementations
  may not have.

  Inputs:
    mu          - the means of the Gaussian models.
    sigma       - the variance of the Gaussian models.
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


from dataclasses import dataclass
import numpy as np
import h5py

from detector.inImage import inImage
from detector.Configuration import AlgConfig

@dataclass
class SGMstate:
  bgIm  : np.ndarray

@dataclass
class SGMdebug:
  mu    : np.ndarray
  sigma : np.ndarray
  errIm : np.ndarray


class CfgSGM(AlgConfig):
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
      init_dict = CfgSGM.get_default_settings()

    super(CfgSGM,self).__init__(init_dict, key_list, new_allowed)

    #print('Test out:')
    #print(self)
    #print(dict(self))
    #print(self.dump())
    #cpy = CfgNode()
    #cpy.deepcopy(self)
    #print(convert_to_dict(cpy))
    #print('--------')

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

    default_dict = dict(tauSigma = 3.5, minSigma = [50.0], alpha = 0.05, \
                        adaptall = False,
                        init = dict( sigma = [20.0] , imsize = None)  )
    return default_dict

  #========================== builtForLearning =========================
  #
  #
  @staticmethod
  def builtForLearning():
    learnCfg = CfgSGM();
    learnCfg.alpha = 0.10
    learnCfg.minSigma = [4.0]
    return learnCfg

  #========================== builtForDepth435 =========================
  #
  #
  @staticmethod
  def builtForDepth435():
    depth_dict = dict(tauSigma = 2.0, minSigma = [0.00009], alpha = 0.05, \
                        adaptall = False,
                        init = dict( sigma = [0.0001] , imsize = None)  )
    learnCfg = CfgSGM(depth_dict);
    return learnCfg

#================================ Gaussian ===============================

class Gaussian(inImage):

  #========================= Gaussian/__init__ =========================
  #
  #
  def __init__(self, bgCfg = None, processor = None, bgMod = None):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    @param[in]  bgMod   The model or parameters for the detector.
    @param[in]  bgCfg   The model or parameters for the detector.
    '''
    super(Gaussian, self).__init__(processor)

    # First, set the configuration member field.
    if bgCfg is None:
      self.config = CfgSGM()
    else:
      self.config = bgCfg

    # Set all runtime member variables and working memory.

    # Last measurement.
    self.measI = None

    #== Background model.
    # If background model passed in, set it.
    # Otherwise set to none.  If model is given in the configuration,
    # then it will be set during _setsize_ invocation.
    #
    if bgMod is not None:
      self.mu    = bgMod.mu
      self.sigma = bgMod.sigma
    else:
      self.mu    = None
      self.sigma = None

    # Working memory.
    self.errI  = None
    self.sqeI  = None
    self.nrmE  = None
    self.maxE  = None
    self.bgI   = None

    # Image dimensions
    self.imsize = None
    if (self.config.init.imsize is not None):
      self._setsize_(self.config.init.imsize)

    # Check for image processor routine.
    self.improcessor = processor


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
    linShape = ( np.prod(self.imsize[0:2]), 1 )

    self.measI = np.zeros( bigShape )
    self.errI  = np.zeros( bigShape )
    self.sqeI  = np.zeros( bigShape )
    self.nrmE  = np.zeros( bigShape ) 
    self.maxE  = np.zeros( linShape ) 
    self.bgI   = np.zeros( linShape , dtype=bool) 

    if (self.sigma is None):
      self.sigma = np.full( bigShape , self.config.init.sigma )

    #if (self.mu is None) && (self.config.init.mu is not None):
    #  self.mu = np.full( bigShape, self.config.init.mu)
    #  NOT IMPLEMENTED.  NEED TO THINK IT THROUGH.
  
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
  def measure(self, I):
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

    if self.mu is None:
      self.mu = self.measI.copy()
      
    # sqeI = (mu - measI).^2 / sigma  (in Matlab paraphrasing).
    # Apply operations wih broadcasting to avoid memory re-allocation.
    # Store outcomes since they get used in the adapt routine.
    np.subtract( self.mu, self.measI, out=self.errI )
    np.square  ( self.errI , out=self.sqeI )
    np.divide  ( self.sqeI , self.sigma, out=self.nrmE )

    # Find max error across dimensions if there are more than 1,
    # as would occur for an RGB image.
    #
    if (self.imsize[2] > 1):
      np.amax( self.nrmE, axis=1, out=self.maxE )
    else:
      np.copyto(self.maxE, self.nrmE )

    np.less( self.maxE, self.config.tauSigma, out=self.bgI )
  
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

    if not self.config.adaptall:                # Get foreground pixels.
      fginds   = np.nonzero(~self.bgI);                          
      oldmu    = self.mu[fginds,:];             # Save current values. 
      oldsigma = self.sigma[fginds,:];

    # Update mean and variance. @todo NEED TO FIX.
    # mu = (1 - alpha) mu + alpha * y = mu - alpha*(mu - y)
    #
    np.subtract( self.mu    , self.config.alpha*self.errI, out=self.mu    )

    # sigma = (1 - alpha) sigma + alpha * (mu - y)^2
    #
    np.multiply( self.sigma , (1-self.config.alpha), out=self.sigma )
    np.multiply( self.sqeI  , self.config.alpha    , out=self.sqeI  )
    np.add( self.sigma, self.sqeI , out=self.sigma )

    # Impose min sigma constraint.
    np.maximum(self.sigma, self.config.minSigma, out=self.sigma)
  
    if not self.config.adaptall:                # Revert foreground values.
      self.mu[fginds,:]    = oldmu      
      self.sigma[fginds,:] = oldsigma

  #============================== process ==============================
  #
  def process(self, I):
    '''!
    @brief  Given a new measurement, apply entire BG modeling pipeline.

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

    cState = SGMstate(bgIm = self.bgI.reshape(self.imsize[0:2]))
    return cState

  #============================== getDebug =============================
  #
  def getDebug(self):

    cDebug = SGMdebug(mu = self.mu.reshape(self.imsize), 
                      sigma = self.sigma.reshape(self.imsize), 
                      errIm = self.maxE.reshape(self.imsize[0:2]))
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

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    wsds = fPtr.create_group("bgmodel.Gaussian")

    wsds.create_dataset("mu", data=self.mu)
    wsds.create_dataset("sigma", data=self.sigma)

    self.config.init.imsize = self.imsize.tolist()
    configStr = self.config.dump()
    wsds.create_dataset("configuration", data=configStr)


  #============================== saveCfg ============================== 
  #
  def saveCfG(self, outFile): # Save to YAML file.
    '''!
    @brief  Save current instance to a configuration file.
    '''
    with open(outFile,'w') as file:
      file.write(self.config.dump())
      file.close()


  #================================ load ===============================
  #
  @staticmethod
  def load(fileName):
    # IAMHERE - [X] Very close to having the load work.
    #           [X] Right now just confirmed recovery of core information.
    #           [X] Next step is to create an onWorkspace instance from the info.
    #           [_] Final step is to run and demonstrate correct loading.
    #
    fptr = h5py.File(fileName,"r")

    gptr = fptr.get("bgmodel.Gaussian")

    muPtr    = gptr.get("mu")
    sigmaPtr = gptr.get("sigma")

    bgMod = SGM.SGMdebug
    bgMod.mu    = np.array(muPtr)
    bgMod.sigma = np.array(sigmaPtr)

    cfgPtr   = gptr.get("configuration")
    configStr = cfgPtr[()].decode()

    fptr.close()

    configCfg = CfgSGM.load_cfg(configStr)

    theConfig = CfgSGM()
    theConfig.merge_from_other_cfg(configCfg)

    theModel = Gaussian(theConfig, None, bgMod)

    return theModel


  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fileName):
    pass


#
#================================ Gaussian ===============================
