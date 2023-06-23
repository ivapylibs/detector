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

@dataclass
class detectorState:
  x: any = None

# @classf detector
class inImage(object):

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
    state = detectorState(self.Ip)
    return state

#
#=========================== detector/inImage ============================
