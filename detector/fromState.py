#=========================== detector/fromState ============================
##
# @package  fromState
# @brief    Detector based on state vector input.
#
# @ingroup  Detector
#
#=========================== detector/fromState ============================

# 
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/08/10          [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
#=========================== detector/fromState ============================


import numpy as np
import h5py

from dataclasses import dataclass

from detector.base import Base, DetectorState


#---------------------------------------------------------------------------
#=========================== detector/fromState ============================
#---------------------------------------------------------------------------


class fromState(Base):
  """!
  @ingroup  Detector
  @brief    Basic detector from state vector input. Should be overloaded, but
            simple detectors can be created with the right pre/post processor.
  """

  #============================== fromState ==============================
  #
  def __init__(self, processor=None):
    '''!
    @brief    Constructor for fromState instance.

    @param[in]    processor   A state processor.
    '''
    self.processor = processor      #< Pre-processor or similar. [partially implemented]
    self.z = None                   #< The detection outcome.

    # @todo     Resolve x vs z as internal detection estimate.

  #=============================== predict ===============================
  #
  def predict(self):
    """!
    @brief  Predict detection state outcome, if implemented.

    Default implementation is to do nothing.  State not updated.
    """
    pass

  #=============================== measure ===============================
  #
  def measure(self, x):
    """!
    @brief  Generate detection outcome from state signal.

    @param[in]  x   External signal to detect with.
    """
    if self.processor is not None:
      self.z = self.processor.apply(x)
    else:
      raise Exception('Processor has not been initialized yet')

  #=============================== correct ===============================
  #
  def correct(self):
    """!
    @brief  Correct detection state estimate, if implemented.

    Default implementation is to do nothing.  Roll with measurement.
    """
    pass

  #================================ adapt ================================
  #
  def adapt(self):
    """!
    @brief  Adapt detection model, if implemented.

    Default implementation is no adaptation.
    """
    pass

  #============================== fromState ==============================
  #
  def detect(self, x):
    """!
    @brief  Perform detection only, which basically keeps the model static
            if it would normally update.

    Detection consists of predict, measure, and correct. No adaptation.  At least
    if there is a measurement.  If no measurement, then only predict is executed
    since there is no measurement to interpret and correct.
    """
    self.predict()
    if x.haveObs:
      self.measure(x.tMeas)
      self.correct()

  #=============================== process ===============================
  #
  def process(self, x):
    """!
    @brief  Run entire processing pipeline.

    The entire pipeline consists of predict, measure, correct, and adapt. At least
    if there is a measurement.  If no measurement, then only predict is executed
    since there is no measurement to interpret, correct, and adapt with.
    """
    self.predict()
    if x.haveObs:
      self.measure(x.tMeas)
      self.correct()
      self.adapt()

  #=============================== getState ==============================
  #
  def getState(self):
    """!
    @brief  Get the current detection state estimate.
    """
    state = DetectorState(x=self.z)
    return state

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    """!
    @brief  Save configuration or other data to HDF5 file.

    @note   Currently not implemented.  Should not get here. Triggers warning.
    """

    warning("Detector.fromState:saveTo -- Should not be here!")
    # Not sure what goes here.  Leaving empty.
    # Maybe eventually save the info strings / structure / dict.

#
#=========================== detector/fromState ============================
