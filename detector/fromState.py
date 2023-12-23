#=========================== detector/fromState ============================
#
#
#=========================== detector/fromState ============================

# 
# @file     fromState.py
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
from dataclasses import dataclass
from detector.base import Base

@dataclass
class detectorState:
  x: any = None

class fromState(Base):
  """!
  @ingroup  Detector
  @brief    Basic detector from state vector input. Should be overloaded, but
            simple detectors can be created with the right pre/post processor.
  """
  def __init__(self, processor=None):
    '''!
    @brief    Constructor for fromState instance.

    @param[in]    processor   A state processor.
    '''
    if isinstance(processor, basic):
      self.processor = processor
    else:
      self.processor = None

    self.z = None

  def predict(self):
    pass

  def measure(self, x):
    if self.processor is not None:
      self.z = self.processor.apply(x)
    else:
      raise Exception('Processor has not been initialized yet')

  def correct(self):
    pass

  def adapt(self):
    pass

  def detect(self, x):
    self.predict()
    self.measure(x)
    self.correct()

  def process(self, x):
    self.predict()
    self.measure(x)
    self.correct()
    self.adapt()

  def getState(self):
    state = detectorState(x=self.z)
    return state

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    # Not sure what goes here.  Leaving empty.
    # Maybe eventually save the info strings / structure / dict.
    pass

  #================================ save ===============================
  #
  def save(self, fileName):    # Save given file.
    fptr = h5py.File(fileName,"w")
    self.saveTo(fptr);
    fptr.close()
    pass


#
#=========================== detector/fromState ============================
