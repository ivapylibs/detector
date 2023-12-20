#=================================== byRegion ==================================
"""!
@package    byRegion
@brief      Activities defined by regions.  Signal must lie in a region to trigger
            associated state.

Should be a general purpose module that triggers activity state when signal enters
known region/zone with a given label.  Whether these zones are mutually exclusive
(disjoint regions) or not is up to the designer.

For now just coding up bare minimum needed.

@note   Yes, this will be the bare minimum and needs to be expanded based on usage.
        There are many ways to define regions and to utilize them as activity
        states or status flags.  Since I aim to instantiate in one specific way 
        at the momemt, that is what will be coded.
"""
#=================================== byRegion ==================================
"""!
@file       byRegion.py

@author     Patricio A. Vela,       pvela@gatech.edu
@date       2023/12/15

"""
#
# NOTE: 90 columns, 2 space indent, wrap margin at 5.
#
#=================================== byRegion ==================================

import numpy as np
from detector.base import Base
import ivapy.display_cv as cvdisplay
import scipy
import skimage.draw as skidraw
import ivapy.Configuration

#-------------------------------------------------------------------------------
#==================================== Planar ===================================
#-------------------------------------------------------------------------------


class Planar(Base):
  """!
  @brief  Activity detection based on lying in specific planar regions.
  """

  def __init__(self):
    pass

  #============================== process ==============================
  #
  def process(self, signal):
    """
    Process the new income signal
    """
    self.predict()
    self.measure(signal)
    self.correct()
    self.adapt()

  #============================== predict ==============================
  #
  def predict(self):
    return None

  #============================== measure ==============================
  #
  def measure(self, signal):
    return None

  #============================== correct ==============================
  #
  def correct(self):
    return None
  
  #=============================== adapt ===============================
  #
  def adapt(self):
    return None


#-------------------------------------------------------------------------------
#=============================== byRegion.inImage ==============================
#-------------------------------------------------------------------------------

#=================================== inImage ===================================
#

class inImage(Base):
  """!
  @brief  Activity states depend on having signal lying in specific regions of an
          image. Signal is presumably in image pixels.

  The presumption is that the regions are disjoint so that each pixel maps to either
  1 or 0 activity/event states.    The signals can be pixel coordinates (2D) or it
  can be pixel coordinates plus depth or height (3D), but up to the underlying class
  implementation to work out whether that extra coordinate matters and its meaning.

  Chosen not to be a sub-class of inImage.inImage due to the fact that these operate
  differently.  Activity recognition checks for presence of signal location in a
  particular region of the image.  Image-based detection checks for presence of
  target features throughout the entire image.

  @note Can code both disjoint and overlap.  Disjoint = single image region label.
        Overlapping requires list/array of region masks.  Multiple states can
        be triggered at once. State is a set/list then, not a scalar.
  """
  # @todo     Should this class employ a configuration? I think so.
  #           Or maybe just have static factory method using configuration?
  #           Or should the configuration have the factory method? Both?

  #========================= inImage / __init__ ========================
  #
  def __init__(self, imRegions = None):
    super(inImage,self).__init__()

    self.isInit = False
    self.lMax   = 0

    if (imRegions is not None):
      self.setRegions(imRegions)

  #============================ initRegions ============================
  #
  def initRegions(self, imsize):
    """!
    @brief    Initialize regions by providing target image dimensions.
              There will be no regions of interest assigned.

    @param[in]    imsize  The image dimensions/shape.  Only first two important.
    """

    self.imRegions = np.zeros(imsize[0:2], dtype='int')
    self.lMax   = 0
    self.isInit = True


  #============================ emptyRegions ===========================
  #
  def emptyRegions(self):
    """!
    @brief    Delete regions image if exists and return to uninitialized state.
    """

    self.imRegions = None
    self.isInit    = False
    self.lMax   = 0

  #============================ wipeRegions ============================
  #
  def wipeRegions(self):
    """!
    @brief    Clear the regions image.  There will be no regions of interest
              assigned.
    """

    if self.isInit:
      self.imRegions.fill(0)

    self.lMax   = 0

  #============================= setRegions ============================
  #
  def setRegions(self, imRegions):
    """!
    @brief    Provide a label image where the pixel label indicates the
              raw activity label.  Semantic meaning is up to the outer
              scope.

    There is no sanity checking to make sure that the image is properly
    given.  Region image just gets used as is, with label possibilities
    defined by max label value, hence they should be ordered from 1 to
    max label value.

    @param[in]    imRegions   Label-type image.
    """

    self.imRegions = imRegions
    self.lMax      = np.max(imRegions)
    self.isInit    = True

  #========================= addRegionByPolygon ========================
  #
  #
  def addRegionByPolygon(self, regPoly, imsize = None):
    """!
    @brief    Add a region by specifying the polygon boundary.  

    The polygon should be closed.  If it is not closed, then it will be closed
    by appending the first point in the polygon vertex list.  Also, image
    regions should be initialized or the target shape given in imsize.

    The polygon should be column-wise in (x,y) coordinates.

    @param[in]    regPoly The polygon region to add. If None, does nothing.
    @param[in]    imsize  The image dimensions, if not yet initialized (optional)
    """

    if regPoly is None:
      return

    regPoly = np.flipud(regPoly)      # numpy array indexing flipped: (x,y) to (i,j)

    if not self.isInit:
      self.initRegions(imsize)

    if self.isInit:
      if (any(regPoly[:,0] != regPoly[:,-1])):
        regPoly = np.hstack((regPoly, np.array(np.transpose([regPoly[:,-1]]))))

      regMask = skidraw.polygon2mask(np.shape(self.imRegions), np.transpose(regPoly))
      self.addRegionByMask(regMask)

  #========================== addRegionByMask ==========================
  #
  def addRegionByMask(self, regMask):
    """!
    @brief    Provide a masked region to use for defining a new state.
              No check to see if it wipes out existing states.

    If the masked region has no true values, then no new region is
    added.

    @param[in]    regMask     Region mask, should match size of internal image.
    """

    if (np.shape(self.imRegions) == np.shape(regMask)):
      if (regMask.any()):
        self.lMax += 1
        self.imRegions[regMask] = self.lMax

  #============================== measure ==============================
  #
  def measure(self, zsig):
    """
    @brief  Compare signal to expected image region states. 

    @param[in]  zsig  The 2D pixel coords / 3D pixel coords + depth value.
    """
    if not self.isInit:
      self.x = 0
      return

    # Map coordinates takes in (i,j). Map zsig from (x,y) to (i,j).
    zsig   = np.flipud(zsig)
    self.x = scipy.ndimage.map_coordinates(self.imRegions, zsig, order = 0)

  #===================== specifyRegionsFromImageRGB ====================
  #
  def specifyPolyRegionsFromImageRGB(self, theImage):
    """!
    @brief    Given an image, get user input as polygons that define the
              different regions.  If some regions lie interior to others,
              then they should be given after.  Order matters.

    Overrides any existing specification.

    @param[in]    theImage
    """

    self.initRegions(np.shape(theImage))

    polyReg = [[1,1]]
    while polyReg is not None:
      polyReg = cvdisplay.getline_rgb(theImage, isClosed = True)
      self.addRegionByPolygon(polyReg)

    # @todo   Maybe keep visualizing/displaying the regions as more get added.
    #         Right now just closes/opens window.


  #============================= display_cv ============================
  #
  def display_cv(self, ratio = 1, window_name = "Activity Regions"):

    if (self.lMax > 0):
      rho    = 254/self.lMax
      regIm  = self.imRegions*rho
    else:
      regIm = self.imRegions

    cvdisplay.gray(regIm.astype(np.uint8), ratio = ratio, window_name = window_name)

  #---------------------------------------------------------------------
  #------------------------- Save/Load Routines ------------------------
  #---------------------------------------------------------------------

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    
    """!
    @brief  Empty method for saving internal information to HDF5 file.

    Save data to given HDF5 pointer. Puts in root.
    """
    actds = fPtr.create_group("activity.byRegion")

    if (self.imRegions is not None):
      actds.create_dataset("imRegions", data=self.imRegions)


  #============================== loadFrom =============================
  #
  def loadFrom(fPtr, relpath="activity.byRegion"):    
    """!
    @brief  Empty method for loading internal information from HDF5 file.

    Load data from given HDF5 pointer. Assumes in root from current file
    pointer location.
    """
    gptr = fptr.get(relpath)

    keyList = list(fPtr.keys())
    if ("imRegions" in keyList):
      regionsPtr = fPtr.get("imRegions")
      imRegions  = np.array(regionsPtr)
    else:
      imRegions  = None

    theDetector = byRegion(imRegions)

    return theDetector

  #---------------------------------------------------------------------
  #------------------------ Calibration Routines -----------------------
  #---------------------------------------------------------------------

  def calibrateFromPolygonMouseInputOverImageRGB(theImage, theFile, initRegions = None):
    """!
    @brief  Calibrate a region detector by requesting closed polygon input from user.

    Has two modes, one of which is to do so from scratch.  Another does some from
    a pre-existing activity region image (basically a label image). Recall that there
    is no check to see if user input is wiping out a previously existing or
    previously enetered activity region.
    """

    imsize      = np.shape(theImage)
    imRegions   = np.zeros( imsize[0:2] , dtype='int' )

    theDetector = byRegion(imRegions)
    theDetector.specifyPolyRegionsFromImageRGB(theImage)

    theDetector.save(theFile)


#
#=================================== byRegion ==================================
