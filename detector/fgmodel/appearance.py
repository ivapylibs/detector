#========================== detector/appearance ==========================
#
# @brief    The most basic object or instance detector from image
#           input. Really should be overloaded, but simple ones can be
#           created from this class with the right pre/post processor.
#       
#========================== detector/appearance ==========================

# 
# @file     appearance.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/03 [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
#========================== detector/appearance ==========================
import _init_paths
from inImage import image # Is this right? I want detector.inImage and to invoke that way

import numpy as np

# @classf detector.fgmodel
class appearance(image):

  def __init__(self, appMod, fgIm):

    super(appearance, self).__init__() # IS CORRECT?
    self._appMod = appMod       #< The appearance model.
    self.fgIm = fgIm

  def getForeGround(self):
    return self.fgIm

  def getBackground(self):
    return np.max(self.fgIm) - self.fgIm
#
#========================== detector/appearance ==========================
