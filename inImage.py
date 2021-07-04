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
class image(object):
  def __init__(self):
    self.processor = []
    self.Ip = []

  def predict(self):
    

  def measure(self, I):
    if (~isempty(self.preprocessor))
      self.Ip = self.processor.apply(I)


  def correct(self):
  

  def adapt(self):
 

  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()


#
#=========================== detector/inImage ============================
