#=========================== detector/inImage ============================
#
#
#=========================== detector/inImage ============================

# @package  detector
# @module   inImage
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

# @classf detector
class inImage(object):
  """!
  @brief    The most basic object or instance detector from image input. 

  Really this class should be overloaded for most cases, but simple image-based
  detectors can be created from this class with the right pre/post processor.
  The trick is to have the improcessor output a binary image for downstream
  use.
  """

  def __init__(self, processor=None):
    if isinstance(processor, basic):
      self.processor = processor
    else:
      self.processor = None

    self.Ip = None

  def predict(self):
    pass

  def measure(self, I):
    if self.processor is not None:
      self.Ip = self.processor.apply(I)
    else:
      raise Exception('Processor has not been initialized yet')

  def correct(self):
    pass

  def adapt(self):
    pass

  def detect(self, I):
    self.predict()
    self.measure(I)
    self.correct()

  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

  def getState(self):
    state = detectorState(x=self.Ip)
    return state

  #================================ save ===============================
  #
  def save(self, fileName):    # Save given file.
    fptr = h5py.File(fileName,"w")
    self.saveTo(fptr);
    fptr.close()
    pass



#
#=========================== detector/inImage ============================
