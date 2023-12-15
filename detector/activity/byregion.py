#=================================== byregion ==================================
"""!
@brief  Activities defined by regions.  Signal must lie in a region to trigger
        associated state.


Should be a general purpose module that triggers activity state when signal enters
known region/zone with a given label.  Whether these zones are mutually exclusive
(disjoint regions) or not is up to the designer.

For now just coding up bare minimum needed.

@note   Yes, this will be the bare minimum and needs to be expanded based on usage.
        There are many ways to define regions and to utilize them as activity
        states or status flags.  Since I aim to instantiate the reasons in one
        specific way, that is what will be coded.
"""
#=================================== byregion ==================================
"""
@file       byregion.py

@author     Patricio A. Vela,       pvela@gatech.edu
@date       2023/12/15
"""
#=================================== byregion ==================================

from detector.activity import base as Base
import ivapy.display_cv as display

#-------------------------------------------------------------------------------
#==================================== Planar ===================================
#-------------------------------------------------------------------------------


class Planar(Base):
    def __init__(self):
        pass

    def process(self, signal):
        """
        Process the new income signal
        """
        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    def predict(self):
        return None

    def measure(self, signal):
        return None

    def correct(self):
        return None
    
    def adapt(self):
        return None


#-------------------------------------------------------------------------------
#=================================== inImage ===================================
#-------------------------------------------------------------------------------

class inImage(Base):
    """!
    @brief  Activity states depend on having signal lying in specific regions of an
            image. Signal is presumably in image pixels.

    The presumption is that the regions are disjoint so that each pixel maps to either
    1 or 0 activity/event states.    The signals can be pixel coordinates (2D) or it
    can be pixel coordinates plus depth or height (3D), but up to the underlying class
    implementation to work out whether that extra coordinate matters and its
    meaning.

    @note Can code both disjoint and overlap.  Disjoint = single image region label.
          Overlapping requires list/array of region masks.  Multiple states can
          be triggered at once. State is a set/list then, not a scalar.
    """

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
    def addRegionByPolygon(self, regPoly, imsize = None)
      """!
      @brief    Add a region by specifying the polygon boundary.  

      The polygon should be closed.  If it is not closed, then it will be closed
      by appending the first point in the polygon vertex list.  Also, image
      regions should be initialized or the target shape given in imsize.

      @param[in]    regPoly The polygon region to add. If None, does nothing.
      @param[in]    imsize  The image dimensions, if not yet initialized (optional)
      """

      if regPoly is None:
        return

      if not self.isInit:
        self.initRegions(imsize)

      if self.isInit:
        if (any(regPoly[:,0] != regPoly[:,-1])):
          regPoly = append(regPoly, np.transpose([regPoly[:,-1]]))

        regMask = skidraw.polygon2mask(np.shape(self.imRegions), regPoly)
        self.addRegionByMask(regMask)

    #========================== addRegionByMask ==========================
    #
    def addRegionByMask(self, regMask)
      """!
      @brief    Provide a masked region to use for defining a new state.
                No check to see if it wipes out existing states.

      If the masked region has no true values, then no new region is
      added.

      @param[in]    regMask     Region mask, should match size of internal image.
      """

      if (all(np.shape(self.imRegions) == np.shape(regMask))):
        if (any(regMask)):
          self.lMax += 1
          self.imRegions[self.regMask] = self.lMax

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

      self.x = scipy.ndimage.map_coordinate(self.imRegions, np.transpose(zsig), order = 0)

    #===================== specifyRegionsFromImageRGB ====================
    #
    def specifyPolyRegionsFromImageRGB(seld, theImage):
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
        polyReg = display.getline_rgb(theImage, isClosed = True)
        self.addPolyRegion(polyReg)

      # @todo   Maybe keep visualizing/displaying the regions as more get added.
      #         Right now just closes/opens window.



## ADD HDF5 LOAD SUPPORT.
## ADD CALIBRATION SUPPORT.

#
#=================================== byregion ==================================
