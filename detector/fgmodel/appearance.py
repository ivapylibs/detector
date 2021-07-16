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
#           Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2021/07/03 [created]
#           2021/07/10 [modified]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
#========================== detector/appearance ==========================

from detector.inImage import inImage
import numpy as np

# @classf detector.fgmodel
class appearance(inImage):

  def __init__(self, appMod, fgIm):

    super(appearance, self).__init__()
    self._appMod = appMod       #< The appearance model.
    self.fgIm = fgIm

  # =========================== getForeground ===========================
  #
  # @brief  Get the foregound class as binary image.
  #
  # @param[out] fgI     Binary foreground image.
  #
  def getForeGround(self):
    assert np.array_equal(self.fgIm, self.fgIm.astype(bool))
    fgI = self.fgIm
    return fgI

  # =========================== getBackground ===========================
  #
  # @brief  Get the backgound class as binary image.
  #
  # @param[out] bgI     Binary background image.
  #
  def getBackground(self):
    assert np.array_equal(self.fgIm, self.fgIm.astype(bool))
    bgI = ~np.array(self.fgIm).astype('bool')
    return bgI
#
#========================== detector/appearance ==========================
