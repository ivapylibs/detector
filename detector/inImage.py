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

# @classf detector
class inImage(object):

  def __init__(self, processor=None):
    if not processor:
      self.processor = []
    else:
      self.processor = processor

    self.Ip = []

  def predict(self):
    pass

  def measure(self, I):
    if self.processor:
      self.Ip = self.processor.apply(I)
    else:
      raise Exception('Processor has not been initialized yet')

  def correct(self):
    pass

  def adapt(self):
    pass

  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()


#
#=========================== detector/inImage ============================
