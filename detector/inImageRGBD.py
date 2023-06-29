#=========================== detector/inImageRGBD ============================
#
# @brief    The most basic object or instance detector from image
#           input. Really should be overloaded, but simple ones can be
#           created from this class with the right pre/post processor.
#
#=========================== detector/inImageRGBD ============================

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
#=========================== detector/inImageRGBD ============================

import numpy as np

from dataclasses import dataclass
from detector.inImage import inImage

import h5py

@dataclass
class detectorState:
  x: any = None

@dataclass
class ImageRGBD:
  color: any = None
  depth: any = None

# @classf detector
class inImageRGBD(inImage):

  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, processor=None):

    super(inImage,self).__init__(processor)

    self.Id = None


  def measure(self, I):
    if self.processor is not None:
      self.Ip = self.processor.color.apply(I.color)
      self.Id = self.processor.depth.apply(I.depth)
    else:
      raise Exception('Processor has not been initialized yet')

  # @todo   There should be two image processors, one for color, one for depth.
  # @note   Added vanilla code for it, but not tested. Do later. Might crash.



#
#=========================== detector/inImageRGBD ============================
