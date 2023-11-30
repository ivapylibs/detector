#=========================== detector/inImage ============================
#
# @brief    The most basic object or instance detector from image
#           input. Really should be overloaded, but simple ones can be
#           created from this class with the right pre/post processor.
#
#=========================== detector/inImage ============================

# 
# @file     inImage.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/03 [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
#=========================== detector/inImage ============================

import numpy as np

from dataclasses import dataclass
from improcessor.basic import basic

import h5py

@dataclass
class detectorState:
  x: any = None


#================================ inImage ================================
#

# @classf detector
class inImage(object):

  #============================== __init__ =============================
  #
  def __init__(self, processor=None):
    if isinstance(processor, basic):
      self.processor = processor
    else:
      self.processor = None

    self.Ip = None

  #============================== predict ==============================
  #
  def predict(self):
    """!
    @brief  Predict next state from current state.

    Base method employs a static state assumption.  Prediction does
    nothing to state.
    """

    pass

  #============================== measure ==============================
  #
  def measure(self, I):
    """!
    @brief  Generate detection measurements from image input.

    Base method really doesn't compute anything, but will apply image
    processing if an image processor is define.  In this manner, simple
    detection schemes may be implemented by passing the input image through
    the image processor. 
    """

    if self.processor is not None:
      self.Ip = self.processor.apply(I)
    else:
      raise Exception('Processor has not been initialized yet')

  #============================== correct ==============================
  #
  def correct(self):
    """!
    @brief  Correct state based on measurement and prediction states.

    Base method does not have correction.
    """
    pass

  #=============================== adapt ===============================
  #
  def adapt(self):
    """!
    @brief  Update/Adapt any internal parameters based on measurements.

    Base method does not have adaptation.
    """
    pass

  #=============================== detect ==============================
  #
  def detect(self, I):
    """!
    @brief  Run detection only processing pipeline (no adaptation).

    Good to have if there is a calibration scheme that uses adaptation, then
    freezes the parameters during deployment.
    """
    self.predict()
    self.measure(I)
    self.correct()

  #============================== process ==============================
  #
  def process(self, I):
    """!
    @brief  Run full detection pipeline, which includes adaptation.

    """
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

  #============================= emptyState ============================
  #
  def emptyState(self):
    """!
    @brief  Return empty state. Useful if contents needed beforehand.

    """
    state = detectorState
    return state

  #============================== getState =============================
  #
  def getState(self):
    """!
    @brief  Return current/latest state. 

    """
    state = detectorState(x=self.Ip)
    return state

  #============================= emptyDebug ============================
  #
  def emptyDebug(self):
    """!
    @brief  Return empty debug state information. Useful if contents needed
            beforehand.

    """

    return None         # For now. just getting skeleton code going.

  #============================== getDebug =============================
  #
  def getDebug(self):
    """!
    @brief  Return current/latest debug state information. 

    Usually the debug state consists of internally computed information
    that is useful for debugging purposes and can help to isolate problems
    within the implemented class or with downstream processing that may
    rely on assumptions built into this implemented class.
    """

    return None         # For now. just getting skeleton code going.

  #================================ info ===============================
  #
  def info(self):
    """!
    @brief  Provide information about the current class implementation.
    
    Exists for reproducibility purposes.  Usually stores the factory 
    information used to build the current class instance.
    """

    tinfo = dict(name = 'filename', version = '0.1', 
                 date = 'what', time = 'now',
                 CfgBuilder = None)

    return tinfo
    # @todo Need to actually make relevant.  Duplicate what is below.

    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    
    """!
    @brief  Empty method for saving internal information to HDF5 file.

    Save data to given HDF5 pointer. Puts in root.
    """
    # Not sure what goes here.  Leaving empty.
    # Maybe eventually save the info strings / structure / dict.
    pass

  #================================ save ===============================
  #
  def save(self, fileName):    # Save given file.
    """!
    @brief  Outer method for saving to a file given as a string.

    Opens file, preps for saving, invokes save routine, then closes.
    Usually not overloaded.  Overload the saveTo member function.
    """
    fptr = h5py.File(fileName,"w")
    self.saveTo(fptr);
    fptr.close()
    pass


#
#=========================== detector/inImage ============================
