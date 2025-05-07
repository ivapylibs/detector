#=============================== differences ===============================
##
# @package  detector.fgmodel.differences
# @brief    Image differencing approach to foreground object recovery.
#
# Image differencing is not really a background model, nor a foreground model
# approach, as the appearance image sotred consists of both elements.
# Nevertheless, since the objective is to return foreground elements as
# opposed to establishing what is background, it fits more closely with a
# foreground detector.  To disambiguate seen foreground elements over time,
# requires further processing to keep track of things.  Under the special
# case that foreground elements lie in unique parts of the image, this
# implementation also supports cumulative storage of detected differences
# (above the minimum accepted size).
#
# @ingroup  Detector_FGModel
#
# @author   Patricio A. Vela,   pvela@gatech.edu
#
# @date     2025/02/14      
#
# NOTES: set tabstop = 4, indent = 2, 85 columns.
#
#=============================== differences ===============================

import numpy as np
import skimage.morphology as morph
import cv2

from ivapy.Configuration import AlgConfig
from detector.fgmodel.appearance import fgAppearance

#
#-------------------------------------------------------------------------
#===================== Differences Configuration Node ====================
#-------------------------------------------------------------------------
#

class CfgDifferences(AlgConfig):
  '''!
  @brief    Configuration setting specifier for Differences FG model.

  The most basic settings are the image difference tolerance and the minimum
  area of accepted foreground regions. 
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    init_dict   Initial dict to provide.  Leave empty unless have
                            previous instance in dict form.
  @param[in]    key_list    Unsure.
  @param[in]    new_allowed     Flag permitting augmentation of dictionary keys.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgDifferences.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

  #========================= get_default_settings ========================
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for method.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''

    default_dict = dict(tauDiff = 4.0, minArea = 0, doAccum = False)
    return default_dict

  #========================== builtForBlackMat =========================
  #
  #
  @staticmethod
  def builtForLearning():
    matCfg = CfgDifferences()
    matCfg.tauDiff = 10
    matCfg.minArea = 20
    return matCfg

#---------------------------------------------------------------------------
#================================ fgGaussian ===============================
#---------------------------------------------------------------------------
#

class fgDifferences(fgAppearance):
  """!
  @ingroup  Detector
  @brief    Image differences foreground model. 

  Core implementation of image difference-based foreground detection.

  For configuration see CfgDifferences.  

  @note  A note on the improcessor.  If the basic version is used, then it
    performs pre-processing.  If a triple version is used, then the
    mid-processor will perform operations on the detected part rather
    than the default operations.  The mid-processor can be used to test
    out different options for cleaning up the binary data.
  """

  #======================= fgDifferences/__init__ ======================
  #
  #
  def __init__(self, fgCfg = None, processor = None, fgIm = None):
    '''!
    @brief  Constructor for image differences foreground detector.
    
    @param[in]  fgCfg       Detector configuration node.
    @param[in]  processor   Image processing to perform prior to/after processing.
    '''
    super(fgDifferences, self).__init__(processor, fgIm)

    # First, set the configuration member field.
    if fgCfg is None:
      self.config = CfgDifferences()
    else:
      self.config = fgCfg

    super(fgDifferences, self).__init__(None, fgIm)

    # Set all runtime member variables and working memory.
    # Foreground model.  
    self.imsize = None
    self.lastI  = None
    self.measI  = None

    self.fgCnt  = 0
    self.labelI = None

    # Check for image processor routine.
    self.improcessor = processor


  #============================= _setsize_ =============================
  #
  def _setsize_(self, imsize):

    if (np.size(imsize) == 2):
      self.imsize = np.append(imsize, [1])
    else:
      self.imsize = imsize;

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
    @brief      Takes image and compares to existing model for detection.
  
    @param[in]    I   Image to process.
    '''
    if self.improcessor is not None: 
      I = self.improcessor.pre(I)
    
    if self.imsize is None:
        self._setsize_(np.array(np.shape(I)))

    self.measI = np.array(I, dtype=np.uint8, copy=True)

    # Step 2: frame difference
    if (self.lastI is None):
      self.lastI = self.measI
      self.labelI = np.zeros(self.imsize[0:2])
      self.fgIm   = np.zeros(self.imsize[0:2], dtype=bool)
    else:
      diff = cv2.absdiff(I, self.lastI)

      if (self.imsize[2] > 1):          # Use an Linf difference.
        diff = np.max(diff, axis=2) 

      mask = diff > self.config.tauDiff;

      if (M is not None):
        mask = np.logical_and(mask, M)

      if (self.config.minArea > 0):
        morph.remove_small_objects(mask.astype('bool'), self.config.minArea, \
                                                        2, out=mask)

      self.fgIm = mask

      nnz = np.count_nonzero(mask)
      if (nnz > 0) and (self.config.doAccum):
        self.fgCnt = self.fgCnt + 1
        self.labelI = self.labelI + mask * self.fgCnt
      

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
    @brief  Update image differencing model based on recent measurement.

    Most basic version just keeps latext image.
    '''
    self.lastI = self.measI

  #=============================== detect ==============================
  #
  def detect(self, I, M = None):
    '''!
    @brief  Given a new measurement, apply the detection pipeline.

    This only goes so far as to process the image and generate the
    detection, with correction.  There is no adaptation. It should
    be run separately if desired.

    Adaptation is important for differences if the objective is to
    see what is new in future frames, rather than new relative to
    first or initialized frame. Do not run this without other 
    adjustments if the comparison image needs to update in time.
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
