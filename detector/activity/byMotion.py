#=========================== detector/activity/byMotion ============================
"""!
@package    byMotion
@brief      Implements simple activity detectors based on movement (or lack of).
"""
#=========================== detector/activity/byMotion ============================

# 
# @file     byMotion.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/08/10          [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  90 columns. wrap margin at 6.
#!  Tab is set to 4 spaces with conversion to spaces.
#
#=========================== detector/activity/byMotion ============================


import numpy as np
from enum import Enum

from improcessor.basic import basic

from detector.Configuration import AlgConfig
import detector.fromState as detBase

class MoveState(Enum):
  STOPPED = 0
  MOVING  = 1

class TrackState(Enum):
  STOPPED = 0
  MOVING  = 1
  GONE    = 2

#
#-------------------------------------------------------------------------
#=============================== CfgMoving ===============================
#-------------------------------------------------------------------------
#

class CfgMoving(AlgConfig):
  '''!
  @brief  Configuration setting specifier for "motion" detector.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgMoving.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #------------------------- get_default_settings ------------------------
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''

    default_dict = dict(tau = 0.01, Cvel = None)
    return default_dict

  #------------------------ builtForImageTracks ------------------------
  #
  #
  @staticmethod
  def builtForImageTracks():
    trackCfg = CfgSGT();
    trackCfg.tau = 3
    return trackCfg

  
#
#-------------------------------------------------------------------------
#================================ isMoving ===============================
#-------------------------------------------------------------------------
#

# @classf detector.activity
class isMoving(detBase.fromState):

  #=========================== isMoving:init ===========================
  '''!
  @brief    Constructor for isMoving activity instance.

  @param[in]    processor   A state processor.
  '''
  def __init__(self, config=CfgMoving(), processor = None):

    if isinstance(processor, basic):
      self.processor = processor
    else:
      self.processor = None

    self.config = config

    self.z = None

  #------------------------------ measure ------------------------------
  '''!
  @brief    Given input, generate measurement of moving state.

  @param[in]    x   The state input (either only velocity or has velocity).
  '''
  def measure(self, x):
    if self.processor is not None:
      # Return y should represent velocities only (x can be either case).
      y = self.processor.apply(x)   
    else:
      # Input x should represent velocities only.
      y = x

    if np.all(np.abs(y) <= self.config.tau):
      self.z = MoveState.STOPPED
    else:
      self.z = MoveState.MOVING


  #------------------------------ getState -----------------------------
  '''!
  @brief    Get current state of motion detector.
  '''
  def getState(self):
    state = detBase.detectorState(x=self.z)
    return state

  #------------------------------- saveTo ------------------------------
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    # Not sure what goes here.  Leaving empty.
    # Maybe eventually save the info strings / structure / dict.

    # FIXME: NEED TO ADD config YAML.
    pass

  #-------------------------------- save -------------------------------
  #
  def save(self, fileName):    # Save given file.
    fptr = h5py.File(fileName,"w")
    self.saveTo(fptr);
    fptr.close()
    pass

#
#-------------------------------------------------------------------------
#============================ isMovingInImage ============================
#-------------------------------------------------------------------------
#

# @classf detector.activity
class isMovingInImage(isMoving):
  '''!
  @brief    A motion detector for track signals from an image.

  Augments super class with a GONE track state since image-based
  measurements can be missing when the target of interest leaves the
  field of view.
  '''

  #======================== isMovingInImage:init =======================
  '''!
  @brief    Constructor for isMovingInImage activity instance.

  @param[in]    processor   A state processor.
  '''
  def __init__(self, processor=None, config=CfgMoving):

    super(isMovingInImage, self).__init__(processor, config)

    self.z = TrackState.GONE

  #------------------------------ measure ------------------------------
  '''!
  @brief    Given input, generate measurement of moving state.

  @param[in]    x   The state input (either only velocity or has velocity).
  '''
  def measure(self, x):
    if (x is None):
      self.z = TrackState.GONE
      return self.z

    if self.processor is not None:
      # Return y should represent velocities only (x can be either case).
      y = self.processor.apply(x)   
    else:
      # Input x should represent velocities only.
      y = x

    if np.all(np.abs(y) <= self.config.tau):
      self.z = TrackState.STOPPED
    else:
      self.z = TrackState.MOVING

#
#=========================== detector/activity/byMotion ============================
