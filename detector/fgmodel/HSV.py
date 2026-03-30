#================================ fgHSV ===============================
##
# @package  detector.fgmodel.fgHSV
# @brief    HSV foreground modeling for detecting simple objects.
#
# @ingroup  Detector_FGModel
#
# @author   Nihit Agarwal       nagarwal90@gatech.edu
#
# @date     2026/02/17

#
#! NOTES: set tabstop = 4, indent = 2, 85 columns.
#
#================================ fgHSV ===============================
from dataclasses import dataclass
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt

from detector.inImage import fgImage
from ivapy.Configuration import AlgConfig
import camera.utils.display as display
#import ivapy.display_cv as display


@dataclass
class HSVState:
  """!
  @brief    Data class for storing HSV foreground output.
  """
  fgIm  : np.ndarray

@dataclass
class HSVDebug:
  """!
  @brief    Data class for storing HSV fg thresholds.
  """
  lower     : np.ndarray
  upper     : np.ndarray


#
#-------------------------------------------------------------------------
#====================== HSV  Configuration Node ======================
#-------------------------------------------------------------------------
#

class CfgHSV(AlgConfig):
  '''!
  @brief    Configuration setting specifier for HSV FG model.

  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgHSV.get_default_settings()

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

    default_dict = dict(lower=[[120, 120, 70]], upper=[[180, 255, 255]])
    return default_dict


  #=========================== builtForRedGlove ==========================
  #
  @staticmethod
  def builtForRedGlove(minArea = 500, initModel = None):
    learnCfg = CfgHSV()
    learnCfg.lower = [[120, 120, 70], [0, 120, 70]]
    learnCfg.upper = [[180, 255, 255], [10, 255, 255]]

    return learnCfg
  
  #=========================== builtForOrangeGlove ========================
  #
  @staticmethod
  def builtForOrangeGlove(minArea = 500, initModel = None):
    learnCfg = CfgHSV()
    learnCfg.lower = [[0, 100, 100]]
    learnCfg.upper = [[20, 255, 255]]

    return learnCfg

#
#---------------------------------------------------------------------------
#================================ fgHSV ===============================
#---------------------------------------------------------------------------
#

class fgHSV(fgImage):
  """!
  @ingroup  Detector
  @brief    Detector for getting the foreground object by checking
            HSV transformed image for certain color ranges.
  """

  #======================== fgHSV/__init__ ========================
  #
  #
  def __init__(self, fgCfg = None, processor = None):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    '''
    super(fgHSV, self).__init__(processor)

    # First, set the configuration member field.
    if fgCfg is None:
      fgCfg = CfgHSV()
    self.config = fgCfg

    # Set all runtime member variables and working memory.
    self.imsize = None

    # Foreground model.  
    self.upper = np.array(fgCfg.upper)
    self.lower = np.array(fgCfg.lower)

    # Set the processor
    self.improcessor = processor

    # Set the foreground image
    self.fgI = None


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
  def measure(self, I, M = None):
    '''!
    @brief    Takes image and generates the detection result.
  
    @param[in]    I   Image to process.
    '''
    if self.improcessor is not None: 
      I = self.improcessor.pre(I)
    
    hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
    # plt.imshow(hsv)
    # plt.title("HSV transform")
    # plt.show()
    total_mask = None
    for i in range(self.lower.shape[0]):
      mask = cv2.inRange(hsv, self.lower[i, :], self.upper[i, :])
      if total_mask is None:
        total_mask = mask
      else:
        total_mask = cv2.add(total_mask, mask)


    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel) # Removes small dots

    self.fgI =  mask


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
    @breif  Adapt the HSV based detection model parameters.
    '''
    pass

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

    eState = HSVState
    return eState

  #============================= emptyDebug ============================
  #
  def emptyDebug(self):

    eDebug = HSVDebug
    return eDebug

  #============================== getState =============================
  #
  def getState(self):

    cState = HSVState(fgIm = self.fgI)
    return cState

  #============================== getDebug =============================
  #
  def getDebug(self):

    cDebug = HSVDebug(lower=self.lower, upper=self.upper) 
    return cDebug



  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    self.config.lower    = self.lower.tolist()
    self.config.upper = self.upper.tolist()

    fPtr.create_dataset("ForegroundHSVModel", data=self.config.dump())

  #============================ saveConfig =============================
  #
  #
  def saveConfig(self, outFile): 
    '''!
    @brief  Save current instance to a configuration file.
    '''

    self.config.lower    = self.lower.tolist()
    self.config.upper = self.upper.tolist()

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

    theSetup = CfgHSV()
    theSetup.merge_from_file(fileName)
    fgDetector = fgHSV(theSetup)
    return fgDetector

  #============================ buildFromCfg ===========================
  #
  #
  @staticmethod
  def buildFromCfg(theConfig, processor = None):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''

    fgDetector = fgHSV(theConfig, processor)
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
    theModel = fgHSV.loadFrom(fptr)
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
    if ("ForegroundHSVModel" in keyList):
      cfgPtr = fPtr.get("ForegroundHSVModel")
      cfgStr = cfgPtr[()].decode()

      theConfig = CfgHSV.load_cfg(cfgStr)
    else:
      print("No foreground Gaussian model; Returning None.")
      theConfig = None

    theModel = fgHSV(theConfig)
    return theModel



#================================ fgHSV ================================
