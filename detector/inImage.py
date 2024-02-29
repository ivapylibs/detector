#=========================== detector/inImage ============================
##
# @package  inImage
# @brief    Detector based on image input.
#
# @ingroup  Detector
#
#

#=========================== detector/inImage ============================
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
from detector.base import Base

import h5py



#================================ inImage ================================
#
class inImage(Base):
  """!
  @ingroup  Detector
  @brief    The most basic object or instance detector from image input. 

  Really this class should be overloaded for most cases, but simple image-based
  detectors can be created from this class with the right pre/post processor.
  The trick is to have the improcessor output a binary image for downstream
  use.
  """

  #============================== __init__ =============================
  #
  def __init__(self, processor=None):
    if isinstance(processor, basic):
      self.processor = processor
    else:
      self.processor = None

    self.Ip = None


  #============================== measure ==============================
  #
  def measure(self, I):
    """!
    @brief  Generate detection measurements from image input.

    Base method really doesn't compute anything, but will apply image
    processing if an image processor is define.  In this manner, simple
    detection schemes may be implemented by passing the input image through
    the image processor. 
    """

    if self.processor is not None:
      self.Ip = self.processor.apply(I)
    else:
      raise Exception('Processor has not been initialized yet')


  #================================ info ===============================
  #
  def info(self):
    """!
    @brief  Provide information about the current class implementation.
    
    Exists for reproducibility purposes.  Usually stores the factory 
    information used to build the current class instance.
    """

    tinfo = dict(name = 'filename', version = '0.1', 
                 date = 'what', time = 'now',
                 CfgBuilder = None)

    return tinfo
    # @todo Need to actually make relevant.  Duplicate what is below.

    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

#================================ fgImage ================================
#

class fgImage(inImage):
  """!
  @ingroup  Detector
  @brief    The most basic object or instance detector from image input. 

  Really this class should be overloaded for most cases, but simple image-based
  detectors can be created from this class with the right pre/post processor.
  The trick is to have the improcessor output a binary image for downstream
  use.
  """

  #============================== __init__ =============================
  #
  def __init__(self, processor=None):

    super(fgImage,self).__init__(processor)

#================================ bgImage ================================
#

class bgImage(inImage):
  """!
  @ingroup  Detector_BGModel
  @brief    The most basic object or instance detector from image input. 

  Image data is considered to be single modality, which typically means 
  grayscale, color (RGB), or depth only imagery.  See bgImageRGBD for color+depth
  modality imagery. Really this class should be overloaded for most cases, but
  simple image-based detectors can be created from this class with the right
  pre/post processor.  The trick is to have the improcessor output a binary image
  for downstream use.
  """

  #============================== __init__ =============================
  #
  def __init__(self, processor=None):

    super(bgImage,self).__init__(processor)



#
#=========================== detector/inImage ============================
