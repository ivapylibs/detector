#=============================== detector/appearance ===============================
#
# @package  detector.fgmodel.appearance
#
# Appearance-based object detection from image stream. Augment basic approach with
# an appearance model that aims to identify foreground regions of the image
#
# @ingroup  Detector_FGModel
#   
# @author   Patricio A. Vela,   pvela@gatech.edu
# @author   Yunzhi Lin,             yunzhi.lin@gatech.edu
#
# @date     2021/07/10 [modified]
# @date     2021/07/03 [created]
#
# NOTE:
#   Indent is 2 spaces.
#   Tab is 4 spaces with conversion.
#   80 columns with margin at 6
#
#=============================== detector/appearance =============================== 
from detector.inImage import fgImage
from detector.base    import DetectorState
import numpy as np

class fgAppearance(fgImage):
  """!
  @ingroup  Detector
  @brief    Appearance based object detection.

  Appearance-based object detection from image stream. Augment basic approach with
  an appearance model that aims to identify foreground regions of the image

  """

  def __init__(self, appMod, fgIm):

    super(fgAppearance, self).__init__()
    self._appMod = appMod       #< The appearance model.
    self.fgIm = fgIm

  #================================ getForeground ================================
  #
  # @brief  Get the foregound class as binary image.
  #
  # @param[out] fgI     Binary foreground image.
  #
  def getForeGround(self):
    assert np.array_equal(self.fgIm, self.fgIm.astype(bool))
    fgI = self.fgIm
    return fgI

  #================================ getBackground ================================
  #
  # @brief  Get the backgound class as binary image.
  #
  # @param[out] bgI     Binary background image.
  #
  def getBackground(self):
    assert np.array_equal(self.fgIm, self.fgIm.astype(bool))
    bgI = ~np.array(self.fgIm).astype('bool')
    return bgI

  #================================== emptyState =================================
  #
  # @brief      Returns the detection mask.
  #
  def emptyState(self):
    state = DetectorState()
    return state

  #=================================== getState ==================================
  #
  # @brief      Returns the detection mask.
  #
  def getState(self):
    state = DetectorState(self.fgIm)
    return state

  #============================= displayForeground_cv ============================
  #
  # @brief  Display the foreground mask using OpenCV routine.
  #
  def displayForeground_cv(self, ratio = None, window_name = "Foreground"):

    import ivapy.display_cv as display
    display.binary(self.fgIm, ratio, window_name)

#
#================================ detector/appearance ================================
