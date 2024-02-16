#=================================== byRegion ==================================
##
# @package  byRegion
# @brief    Activities defined by regions.  Signal must lie in a region to trigger
#           associated state.
# 
# Should be a general purpose module that triggers activity state when signal enters
# known region/zone with a given label.  Whether these zones are mutually exclusive
# (disjoint regions) or not is up to the designer.
# 
# For now just coding up bare minimum needed.
# 
# @ingroup  Detector_Activity
# @note     Yes, this will be the bare minimum and needs to be expanded based on usage.
#           There are many ways to define regions and to utilize them as activity
#           states or status flags.  Since I aim to instantiate in one specific
#           way at the momemt, that is what will be coded.

#=================================== byRegion ==================================
# 
# @author     Patricio A. Vela,       pvela@gatech.edu
# @date       2023/12/15
# 
# NOTE: 90 columns, 2 space indent, wrap margin at 5.
#
#=================================== byRegion ==================================


#======================== Environment / API Dependencies =======================
import numpy as np
import h5py

from detector.base import Base
from detector.fromState import fromState
import ivapy.display_cv as cvdisplay
import scipy
import scipy.ndimage
import skimage.draw as skidraw
import ivapy.Configuration

#-------------------------------------------------------------------------------
#==================================== Planar ===================================
#-------------------------------------------------------------------------------


class Planar(fromState):
  """!
  @ingroup  Detector_Activity
  @brief    Activity detection based on lying in specific planar regions.
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
#=============================== byRegion.imageRegions ==============================
#-------------------------------------------------------------------------------

# @todo     Eventually may want to create configuration class that specifies how
#           to build an imageRegions instance. Parses fields is particular order
#           and reconstructs based on field entries.  
#

#=================================== imageRegions ===================================
#

class imageRegions(fromState):
  """!
  @ingroup  Detector_Activity
  @brief    Activity states depend on having signal lying in specific regions of an
            image. Signal is presumably in image pixels.

  The presumption is that the regions are disjoint so that each pixel maps to either
  1 or 0 activity/event states.    The signals can be pixel coordinates (2D) or it
  can be pixel coordinates plus depth or height (3D), but up to the underlying class
  implementation to work out whether that extra coordinate matters and its meaning.

  Chosen not to be a sub-class of inImage due to the fact that these operate
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

  #========================= imageRegions / __init__ ========================
  #
  def __init__(self, imRegions = None):
    super(imageRegions,self).__init__()

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
    added.  If multiple slices of masks are given (3rd dimension), then
    will add each slice.

    @param[in]    regMask     Region mask, should match size of internal image.
    """

    regSize = np.shape(regMask)

    if (np.shape(self.imRegions) == regSize[0:2]):
      if (len(regSize) == 3):
        for ii in range(0,regSize[2]):
          self.addRegionByMask(regMask[:,:,ii])
      else:
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
    zmeas  = np.flipud(zsig.tMeas)
    self.x = scipy.ndimage.map_coordinates(self.imRegions, zmeas, order = 0)

  #===================== specifyRegionsFromImageRGB ====================
  #
  def specifyPolyRegionsFromImageRGB(self, theImage, doClear = False):
    """!
    @brief    Given an image, get user input as polygons that define the
              different regions.  If some regions lie interior to others,
              then they should be given after.  Order matters.

    Overrides any existing specification.

    @param[in]  theImage    The source image to provide region context.
    @param[in]  doClear     Optional: clear existing imregions?
    """

    if (doClear) or (self.imRegions is None) \
                 or (np.shape(theImage)[0:2] != np.shape(self.imRegions)):

      self.initRegions(np.shape(theImage))

    polyReg = [[1,1]]
    while polyReg is not None:
      polyReg = cvdisplay.getline_rgb(theImage, isClosed = True)
      self.addRegionByPolygon(polyReg)

    # @todo   Maybe keep visualizing/displaying the regions as more get added.
    #         Right now just closes/opens window.


  #============================= printState ============================
  #
  def printState(self):

    print("State: " + str(self.x))

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
  def saveTo(self, fPtr, relpath="activity.byRegion"):    
    """!
    @brief  Empty method for saving internal information to HDF5 file.

    Save data to given HDF5 pointer. Puts in root.

    @param[in]  fPtr        HDF5 file pointer.
    @param[in]  relpath     Group name relative to current HDF5 path.
    """
    actds = fPtr.create_group(relpath)

    if (self.imRegions is not None):
      actds.create_dataset("imRegions", data=self.imRegions)


  #================================ load ===============================
  #
  @staticmethod
  def load(fileName, relpath = None):    # Load given file.
    """!
    @brief  Outer method for loading file given as a string (with path).

    Opens file, preps for loading, invokes loadFrom routine, then closes.
    Overloaded to invoke coorect loadFrom member function.

    @param[in]  fileName    The full or relative path filename.
    @param[in]  relpath     The hdf5 (relative) path name to use for loading.
                            Usually class has default, this is to override.
    """
    fptr = h5py.File(fileName,"r")
    if relpath is not None:
      theInstance = imageRegions.loadFrom(fptr, relpath);
    else:
      theInstance = imageRegions.loadFrom(fptr)
    fptr.close()
    return theInstance

  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fptr, relpath="activity.byRegion"):    
    """!
    @brief  Empty method for loading internal information from HDF5 file.

    Load data from given HDF5 pointer. Assumes in root from current file
    pointer location.
    """
    gptr = fptr.get(relpath)

    keyList = list(gptr.keys())
    if ("imRegions" in keyList):
      regionsPtr = gptr.get("imRegions")
      imRegions  = np.array(regionsPtr)
    else:
      imRegions  = None

    theDetector = imageRegions(imRegions)

    return theDetector

  #---------------------------------------------------------------------
  #------------------------ Calibration Routines -----------------------
  #---------------------------------------------------------------------

  #============= calibrateFromPolygonMouseInputOverImageRGB ============
  #
  def calibrateFromPolygonMouseInputOverImageRGB(theImage, theFile, initRegions = None):
    """!
    @brief  Calibrate a region detector by requesting closed polygon input from user.

    Calibration routines save the information for loading in the future.  They
    do not return an instantiated object.

    This version has two modes, one of which is to specify from scratch.  Another
    takes a pre-existing activity region image (basically a label image).  Recall
    that there is no check to see if user input is wiping out a previously
    existing or previously enetered activity region.
    """

    imsize      = np.shape(theImage)
    if (initRegions is not None) and (imsize[0:2] == np.shape(initRegions)):
      pass
    else:
      initRegions = np.zeros( imsize[0:2] , dtype='int' )

    theDetector = imageRegions(initRegions)
    theDetector.specifyPolyRegionsFromImageRGB(theImage)

    theDetector.display_cv();
    cvdisplay.wait()

    theDetector.save(theFile)

  #---------------------------------------------------------------------
  #----------------------- Factory/Build Routines ----------------------
  #---------------------------------------------------------------------

  def buildFromPolygons(imsize, thePolygons):
    """!
    @brief  Construct an imageRegions instance with provided polygon regions.

    @param[in]  imsize      The image size.
    @param[in]  thePolygons List of polygons as column array of coordinates.

    @return     Instantiated object.
    """

    actDet = imageRegions(np.zeros(imsize))
    for poly in thePolygons:
      actDet.addRegionByPolygon(poly)

    return actDet

#
#=================================== byRegion ==================================
