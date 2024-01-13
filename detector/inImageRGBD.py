#=========================== detector/inImageRGBD ============================
##
# @package  inImageRGBD
# @brief    Detector based on RGBD imagery.
# @ingroup  Detector
#
#=========================== detector/inImageRGBD ============================

# 
# @file     inImageRGBD.m
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
from camera.base import ImageRGBD

import h5py

class inImageRGBD(inImage):
  """!
  @ingroup  Detector
  @brief    Most basic object or instance detector from RGBD image input. 

  This class should be overloaded for most cases. Simple based detectors can be
  created from this class with the right pre/post processor if the first
  improcessor pass (_pre_) generates a binary image for downstream use.

  It differs from the standard inImage version inthat the input consists of
  two packaged streams, and RGB stream and a D stream.  The channels are
  not concatenated since their types differ.

  @todo Should it really be a sub-class of inImage?
  """

  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, processor=None):

    super(inImageRGBD,self).__init__(processor)

    self.Id = None


  def measure(self, I):
    if self.processor is not None:
      self.Ip = self.processor.color.apply(I.color)
      self.Id = self.processor.depth.apply(I.depth)
    else:
      raise Exception('Processor has not been initialized yet')

  # @todo   There should be two image processors, one for color, one for depth.
  # @note   Added vanilla code for it, but not tested. Do later. Might crash.


#============================== bgImageRGBD ==============================
#

class bgImageRBGD(inImageRGBD):
  """!
  @ingroup  Detector_BGModel
  @brief    The most basic object or instance detector from RGBD image input. 

  Really this class should be overloaded for most cases, but simple image-based
  detectors can be created from this class with the right pre/post processor.
  The trick is to have the improcessor output a binary image for downstream
  use.
  """

  #============================== __init__ =============================
  #
  def __init__(self, processor=None):

    super(bgImageRGBD,self).__init__(processor)



#
#=========================== detector/inImageRGBD ============================
