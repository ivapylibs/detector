#============================== onWorkspace ==============================
"""
  @class Gaussian

  @brief    Applies Gaussian model with premise that targets lie to one side of
            the distribution.

  Background estimation/learning still uses a Gaussian model, but the detection
  part only checks for the error in one direction (by not squaring the error
  quantity).  This version makes sense for depth cameras as objects on the
  workspace will be closer to the camera (at least when viewed from above down
  towards the workspace). Naturally, there is a static model assumption going
  on here.

  Inputs:
    mu          - the means of the Gaussian models.
    sigma       - the variance of the Gaussian models.
    weights     - the weights of the Gaussian models.
    parms       - [optional] configuration instance with parameters specified.

  Fields of the parms structure:
    sigma       - Initial variance to use if sigma is empty.
    thresh      - Threshold for determining foreground.
    alpha       - Update rate for mean and variance.
    lambdaSigma - Update rate scaling for variance (defult = 1). 
                  Usually it is best to update more slowly.

    A note on the improcessor.  If the basic version is used, then it
    performs pre-processing.  If a triple version is used, then the
    mid-processor will perform operations on the detected part rather
    than the default operations.  The mid-processor can be used to test
    out different options for cleaning up the binary data.
"""
#============================== onWorkspace ==============================
#
# @file     Gaussian.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/15      [converted to python]
#
# Version   1.0
#
# Notes:    set tabstop = 4, indent = 2, 85 columns.
#
#============================== onWorkspace ==============================

from enum import Enum
from dataclasses import dataclass

import numpy as np
import h5py
import cv2

import camera.utils.display as display

from detector.inImage import inImage
import detector.bgmodel.Gaussian as SGM


class RunState(Enum):
  ESTIMATE = 1
  DETECT   = 2
    
class CfgOnWS(SGM.CfgSGM):
  '''!
  @brief  Configuration setting specifier for Gaussian workspace model.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgOnWS.get_default_settings()

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

    default_dict = SGM.CfgSGM.get_default_settings()
    #append_dict  = dict(NOTHING FOR NOW)
    #default_dict.update(append_dict)

    return default_dict

  #========================== builtForDepth435 =========================
  #
  #
  @staticmethod
  def builtForDepth435():
    '''!
    @brief  On Workspace model parameters for Realsense 435 mounted
            a given distance from tabletop, looking down.

    '''
    depth_dict = dict(tauSigma = 1.5, minSigma = [0.0002], alpha = 0.05, \
                        lambdaSigma = 1, adaptall = True,
                        init = dict( sigma = [0.005] , imsize = [])  )
    learnCfg = CfgOnWS(depth_dict);
    return learnCfg

  #========================= builtForPuzzlebot =========================
  #
  #
  @staticmethod
  def builtForPuzzlebot():
    '''!
    @brief  On Workspace model parameters for Realsense 435 mounted
            a given distance from tabletop, looking down.

    '''
    depth_dict = dict(tauSigma = 2.5, minSigma = [0.0001], alpha = 0.025, \
                        lambdaSigma = 0.3, adaptall = True,
                        init = dict( sigma = [0.001] , imsize = [])  )
    learnCfg = CfgOnWS(depth_dict);
    return learnCfg


#
#---------------------------------------------------------------------------
#=============================== onWorkspace ===============================
#---------------------------------------------------------------------------
#

class onWorkspace(SGM.Gaussian):

  #========================= Gaussian/__init__ =========================
  #
  #
  def __init__(self, bgCfg = None, processor = None, bgMod = None):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    @param[in]  bgMod   The model or parameters for the detector.
    @param[in]  bgCfg   The model or parameters for the detector.
    '''
    super(onWorkspace, self).__init__(bgCfg, processor, bgMod)

    onWorkspace.state    = RunState.ESTIMATE
    onWorkspace.tauStDev = np.sqrt(self.config.tauSigma)


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
    
    if (self.imsize is None) or (len(self.imsize) == 0):
        self._setsize_(np.array(np.shape(I)))

    self.measI = np.array(I, dtype=float, copy=True)
    display.depth_cv(self.measI)
    self.measI = np.reshape(self.measI, 
                            np.append(np.prod(self.imsize[0:2]), self.imsize[2]) )

    if self.mu is None:
      self.mu = self.measI.copy()
      
    np.subtract( self.mu, self.measI, out=self.errI )

    if (self.state == RunState.ESTIMATE):   # Standard Gaussian modeling.
      # sqeI = (mu - measI).^2 / sigma  (in Matlab paraphrasing).
      # Apply operations wih broadcasting to avoid memory re-allocation.
      # Store outcomes since they get used in the adapt routine.
      np.square  ( self.errI , out=self.sqeI )
      np.divide  ( self.sqeI , self.sigma, out=self.nrmE )

      # Find max error across dimensions if there are more than 1,
      # as would occur for an RGB image. While the premise underlying
      # this class is that there is a depth camera, such flexibility
      # is still afforded just in case.
      #
      if (self.imsize[2] > 1):
        np.amax( self.nrmE, axis=1, out=self.maxE )
      else:
        np.copyto(self.maxE, self.nrmE )

      np.less( self.maxE, self.config.tauSigma, out=self.bgI )
    else:
      # Repurpose sqeI as the square root of sigma to get st dev.
      np.sqrt   ( self.sigma, out=self.sqeI )
      np.divide ( self.errI, self.sqeI, out=self.nrmE )

      if (self.imsize[2] > 1):
        np.amax( self.nrmE, axis=1, out=self.maxE )
      else:
        np.copyto(self.maxE, self.nrmE )

      # Positive values mean that distance is lower. Negative values, further.
      # Why? When looking own at a surface, expect lower distance to be closer
      # to camera and therefore positive as per previous sentence.
      # Only need to apply threshold for "closer" distances, e.g., more positive
      # error values.
      #
      np.less( self.maxE, self.tauStDev, out=self.bgI )

  
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

    if (self.state == RunState.DETECT) or (self.config.alpha == 0):
      return

    if not self.config.adaptall:                # Get foreground pixels.
      fginds   = np.nonzero(~self.bgI);                          
      oldmu    = self.mu[fginds,:];             # Save current values. 
      oldsigma = self.sigma[fginds,:];

    # Update mean and variance. @todo NEED TO FIX.
    # mu = (1 - alpha) mu + alpha * y = mu - alpha*(mu - y)
    #
    np.subtract( self.mu    , self.config.alpha*self.errI, out=self.mu    )

    # sigma = (1 - alphaSigma) sigma + alphaSigma * (mu - y)^2
    #
    alphaSigma = self.config.alpha * self.config.lambdaSigma
    np.multiply( self.sigma , (1-alphaSigma), out=self.sigma )
    np.multiply( self.sqeI  , alphaSigma    , out=self.sqeI  )
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

    cState = SGM.SGMstate(bgIm = self.bgI.reshape(self.imsize[0:2]))
    return cState

  #============================== getDebug =============================
  #
  def getDebug(self):

    cDebug = SGM.SGMdebug(mu = self.mu.reshape(self.imsize), 
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
    wsds = fPtr.create_group("bgmodel.onWorkspace")

    wsds.create_dataset("mu", data=self.mu)
    wsds.create_dataset("sigma", data=self.sigma)

    self.config.init.imsize = self.imsize.tolist()
    configStr = self.config.dump()
    wsds.create_dataset("configuration", data=configStr)

  #======================== estimateOutlierMask ========================
  #
  # @brief  Apply to a stream for a given number of frames, then find pixels
  #         that are intermittently or persistently evaluating to true.
  #         These are unreliable.  Should be masked out.
  #
  # The stream is presumed to be a depth + color stream as obtained from
  # a Realsense camera.  Code is not as generic as could be.
  #
  # @todo   Modify to be a bit more generic.
  #
  # @param[in]  theStream   RGBD stream to use.
  # @param[in]  numLoop     Number of times to loop (or until keyhit <= 0).
  # @param[in]  tauRatio    Ratio to recover outlier threshold count.
  # @param[in]  incVis      Include visualization during process?
  #
  def estimateOutlierMaskRGBD(self, theStream, numLoop = 0, tauRatio = 0.75,
                                                            incVis = False):

    print('\n STEPS to for outlier estimation.')
    print('\t [1] Make sure workspace is empty and have initial BG model.')
    print('\t [2] Hit enter to continue once ready.')

    if (numLoop > 0):
      print('\t     Process will end on its own.')
    else:
      print('\t [3] Hit "q" to stop outlier estimation process. Not too long.')

    input()

    surfaceCount  = None
    loopCount   = 0

    while (loopCount < numLoop) or (numLoop <= 0):
      # @todo   Modify to use the RGBD call to get frame/image. Code is old.
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      self.process(dep)

      bgS = self.getState()

      if (surfaceCount is None):
        surfaceCount = bgS.bgIm.astype('int')
      else:
        surfaceCount += bgS.bgIm.astype('int')

      loopCount+=1

      if (incVis):
        bgD = self.getDebug()

        bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        display.rgb_depth_cv(bgIm, bgD.mu, ratio=0.25, window_name="RGB+Depth")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break

    tauCount = np.floor(1+tauRatio*loopCount)
    outMask = surfaceCount < tauCount

    if (incVis):
      display.close_cv("RGB+Depth")

    return outMask




  #
  #---------------------------------------------------------------------
  #====================== Static Member Functions ======================
  #---------------------------------------------------------------------
  #

  #==================== buildAndCalibrateFromConfig ====================
  #
  # @brief  build and calibrate onWorkspace model from an initial config 
  #         and a camera class streaming camera. Return instantiated and 
  #         calibrated model.
  #         
  # The stream is presumed to be a depth + color stream as obtained from
  # a Realsense camera.  Code is not as generic as could be.
  #
  # @todo   Modify to be a bit more generic.
  #
  @staticmethod
  def buildAndCalibrateFromConfig(theConfig, theStream, incVis = False):

    print('\n STEPS to calibrate onWorkspace.')
    print('\t [1] Make sure workspace is empty.')
    print('\t [2] Hit enter to continue once scene is prepped.')
    print('\t [3] Hit "q" to stop adaptation process. Should be short.')
    input()

    bgModel = onWorkspace( theConfig )
 
    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      bgModel.process(dep)

      if (incVis):
        bgS = bgModel.getState()
        bgD = bgModel.getDebug()

        bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        display.rgb_depth_cv(bgIm, bgD.mu, ratio=0.25, window_name="RGB+Depth")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break

    display.close_cv("RGB+Depth")
    return bgModel



  #================================ load ===============================
  #
  @staticmethod
  def load(fileName):

    fptr = h5py.File(fileName,"r")
    theModel = onWorkspace.loadFrom(fptr)
    fptr.close()

    return theModel

  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fptr):
    gptr = fptr.get("bgmodel.onWorkspace")

    muPtr    = gptr.get("mu")
    sigmaPtr = gptr.get("sigma")

    bgMod = SGM.SGMdebug
    bgMod.mu    = np.array(muPtr)
    bgMod.sigma = np.array(sigmaPtr)

    cfgPtr   = gptr.get("configuration")
    configStr = cfgPtr[()].decode()


    theConfig = CfgOnWS.load_cfg(configStr)
    theModel  = onWorkspace(theConfig, None, bgMod)
    return theModel


  #====================== onWorkspace/buildFromCfg =====================
  #
  @staticmethod
  def buildFromCfg(theConfig, processor=None):
    '''!
    @brief  Build an onWorkspace instance from an algorithm configuration node.

    @param[out] bgDet   Instantiated onWorkspace background model detector.
    '''
  
    #DEBUG
    #print('onWorkspace -----')
    #print(theConfig)
    #print('onWorkspace #####')
    bgDet = onWorkspace(theConfig, processor)
    return bgDet

  

#
#============================== onWorkspace ==============================
