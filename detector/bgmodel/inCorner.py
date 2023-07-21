#============================ detector.fgmodel.inCorner ============================
'''!
  @brief    Class instance for extracting background that lies with the corner
            region of the RGB color cube, or the extremal regions of a
            color-space based on a planar cut.

  The idea is similar to the fgmodel approach based on planar cuts but
  with the presumption that the background is what has the unique color
  profile.  The presumption is that the background lies in a region of 
  color-space that is extremal so that a planar surface separates it from
  the expected foreground colors.
'''
#============================ detector.fgmodel.inCorner ============================
'''!

  @author   Patricio A. Vela,       pvela@gatech.edu
  @date     2023/04/13
'''
# NOTE:
#   Expects 100 character width due to 4-space indents in python.
#   Comment separators also wider to 84 characters rather than 74, with others
#   scaled up too (72 -> 78).
#
#============================ detector.fgmodel.inCorner ============================

import numpy as np
import h5py

import cv2

import camera.utils.display as display

from skimage import measure
from skimage import morphology

from detector.Configuration import AlgConfig
from detector.inImage import inImage

# Struct for tModel
class SurfaceCutModel(object):

    def __init__(self):
      pass

#================================== PlanarModel ==================================
#
#
class PlanarModel(SurfaceCutModel):
  '''!
    @brief    Specification data class that describes the planar cut classifier
              and its functional properties.
  '''
  def __init__(self, n=None, d = None, tau=None, vectorize=True):
      self.n   = n
      self.d   = d
      self.tau = tau
      self.vectorize = vectorize

      self.classify = None
      self.margin   = None
      self._genLambdaFunctions()


  def _genLambdaFunctions(self):
      if (self.d == 0):
        self.classify = lambda c: self.n @ c > self.tau
      else:
        self.classify = lambda c: self.n @ c + self.d < self.tau

      if (self.d == 0):
        self.margin = lambda c: self.n @ c - self.tau
      else:
        self.margin = lambda c: self.n @ c + self.d - self.tau


  def adjustThreshold(self, ntau):
      if (np.isscalar(ntau)):
          self.tau = ntau
      else:
          if (self.vectorize):
              self.tau = np.array(ntau).reshape(-1,1).T
          else:
              self.tau = ntau

      self._genLambdaFunctions()
  
  def offsetThreshold(self, dtau):
    self.tau = self.tau + dtau

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    gds = fPtr.create_group("PlanarModel")
    print("Creating group: PlanarModel")

    gds.create_dataset("n", data=self.n)
    gds.create_dataset("d", data=self.d)
    gds.create_dataset("tau", data=self.tau)
    gds.create_dataset("vectorize", data=self.vectorize)


  #============================ build_model ============================
  #
  @staticmethod
  def build_model(n, d, tau, isVectorized=True):
      '''!
      @brief  Build a model given the arguments as specifications.

      @param[in]  n               Normal vector.
      @param[in]  d               Separating boundary offset.
      @param[in]  tau             Threshold/one-side margin.
      @param[in]  isVectorized    Boolean: operation should be vectorized or not.
                                  Default is True.
      '''

      theModel = PlanarModel(n, d, tau, isVectorized)

      return theModel

  #====================== PlanarModel/buildFromCfg =====================
  #
  @staticmethod
  def buildFromCfg(theParms, tau = 0, isVectorized = True):
    '''!
    @brief  Build a PlanarModel instance from a configuration node.

    @param[out] pModel   Instantiated planar model classifier.
    '''

    pModel = PlanarModel(np.array(theParms.n), np.array(theParms.d),
                                                                tau, isVectorized)
    return pModel

  #=============================== loadFrom ==============================
  #
  @staticmethod
  def loadFrom(fptr):
      # IAMHERE - [_] Work on getting basic load function.
      #           [_] Confirm recovery of core information.
      #           [_] Next step is to create an instance from the info.
      #           [_] Final step is to run and demonstrate correct loading.
      #
      gptr = fptr.get("PlanarModel")

      nPtr = gptr.get("n")
      dPtr = gptr.get("d")
      tPtr = gptr.get("tau")
      vPtr = gptr.get("vectorize")

      n = np.array(nPtr)
      d = np.array(dPtr)
      tau = np.array(tPtr)
      vectorize = np.array("vectorize", dtype=np.bool_)

      theModel = PlanarModel(n, d, tau, vectorize)
      return theModel
  

# @todo This code is out of date.  Needs to look like the PlanarModel code. 
class SphericalModel(SurfaceCutModel):
  '''!
    @brief    Specification data class that describes the planar cut classifier
              and its functional properties.
  '''
  def __init__(self, c=None, r = 0, tau = 0, classify=None, vectorize=True):
      self.c = c
      self.r = r
      self.tau = tau
      self.classify = classify
      self.vectorize = vectorize

  @staticmethod
  def build_model(c, r, tau, isVectorized=True):
      '''!
      @brief  Build a model given the arguments as specifications.

      @param[in]  c               Origin or center of region.
      @param[in]  r               Separating boundary offset (radius of sphere).
      @param[in]  tau             Threshold/one-side margin.
      @param[in]  isVectorized    Boolean: operation should be vectorized or not.
                                  Default is True.
      '''

      theModel = SphericalModel(c, r, tau)

      if (theModel.r == 0):
        theModel.classify = lambda c: np.linalg.norm(np.subtract(c,theModel.c), axis=0) < theModel.tau
      else:
        theModel.classify = lambda c: np.linalg.norm(np.subtract(c,theModel.c), axis=0) - theModel.r < theModel.tau

      theModel.vectorize = isVectorized

      return theModel

  #============================ buildFromCfg ===========================
  #
  @staticmethod
  def buildFromCfg(theParms, tau = 0, isVectorized = True):
    '''!
    @brief  Build a PlanarModel instance from a configuration node.

    @param[out] pModel   Instantiated planar model classifier.
    '''

    pModel = SphericalModel(np.array(theParms.c), np.array(theParms.r),
                                                                   tau, isVectorized)


#
#-----------------------------------------------------------------------------------
#================================== Configuration ==================================
#-----------------------------------------------------------------------------------
#

class CfgInCorner(AlgConfig):
  '''!
  @brief  Configuration setting specifier for Gaussian workspace model.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    init_dict = CfgInCorner.get_default_settings()

    super(CfgInCorner,self).__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #========================= get_default_settings ========================
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

    default_dict = dict(cutModel = 'Planar', 
                        cutParmsPlanar    = dict(n = [0.57735, 0.57735, 0.57735], 
                                                 d = -50.0),
                        cutParmsSpherical = dict(c = [0.0, 0.0, 0.0], r = 80.0), 
                        isVectorized = True,
                        tau = 0.0) 
    #default_dict = dict(blah = 5)

    return default_dict


#
#-----------------------------------------------------------------------------------
#===================================== inCorner ====================================
#-----------------------------------------------------------------------------------
#

class inCorner(inImage):

  #================================ inCorner ===============================
  #
  #
  def __init__(self, processor = None, bgMod = None):
      '''!
      @brief  Constructor for corner color model target FG detector.
  
      @param[in]  appMod  The model or parameters for the appearance detector.
      '''
      super(inCorner, self).__init__(processor)

      self.bgModel = bgMod

  #============================= specify_model =============================
  #
  def specify_model(self, n, d, tau, isVectorized = True):
      '''!
      @brief  Specify the model by providing normal, distance, and threshold parameter(s). 

      The distance offset and threshold operate additively, this it may be the case
      the one is set to zero and the other is not.  Alternatively, it may be more
      intuitive to have them be separate since dist defines the location of the seprating
      plane and tau defines the detection margin/slack associated to the separating plane.
      It all depends on what is more intuitive to the designer.  All options permitted in
      this case.

      @param[in]  n       Normal vector.
      @param[in]  d       Distance offset.
      @param[in]  tau     Threshold.
      '''
      tModel = PlanarModel.build_model(n, d, tau, isVectorized)
      self.set_model(tModel)

      return tModel


  #=============================== set_model ===============================
  #
  #
  def set_model(self, pCutModel):
      '''!
      @brief  Provide the background "classification" model to use for detection. 

      @param[in]  pCutModel   Background planar cut model instance.
      '''
      self.bgModel = pCutModel


  #================================ measure ================================
  #
  #
  def measure(self, I):
      '''!
      @brief  Apply the appearance detector to an image.

      @param[in]  I   Image to test on.
      '''
      if self.processor:
          pI = self.processor.apply(I)
      else:
          pI = I

      if self.bgModel.vectorize:
          imDat   = np.array(pI).reshape(-1,pI.shape[2]).T
          fgB     = self.bgModel.classify(imDat)
          self.Ip = np.array(fgB).reshape(pI.shape[0], pI.shape[1])
      else:
          self.Ip = self.bgModel.classify(pI)

  #============================== calc_margin ==============================
  #
  #
  def calc_margin(self, I):
      '''!
      @brief  Run classifier scoring computation and return value. 

      @param[in]  I   Image to test on.
      @param[out] mI  The margin values (as an image/2D array).
      '''
      if self.processor:
          pI = self.processor.apply(I)
      else:
          pI = I

      if self.bgModel.vectorize:
          imDat = np.array(pI).reshape(-1,pI.shape[2]).T
          fgB   = self.bgModel.margin(imDat)
          mI    = np.array(fgB).reshape(pI.shape[0], pI.shape[1])
      else:
          mI    = self.bgModel.margin(pI)

      return mI


  #
  #---------------------------------------------------------------------------
  #========================= Static Member Functions =========================
  #---------------------------------------------------------------------------
  #


  #========================== build_model_blackBG ==========================
  #
  @staticmethod
  def build_model_blackBG(dist, tau, isVectorized = True):
      '''!
      @brief  Build a black color-based background model. The assumption is that
              the background colors are in the black corner of the color cube
              and can be split from the foreground colors. In this case, the
              planar cut normal is known.

      The good news about this kind of model is that it does not matter whether
      the image is RGB (typical) or BGR (OpenCV type).  The corner region containing 
      black is still around (0,0,0).

      @param[in]  dist    Locate of separating boundary (should be negative). 
      @param[in]  tau     The threshold / margin to apply.
      '''

      n = np.array([1, 1, 1])/np.sqrt(3)      # Points to white color along gray line.
                                              # Dist should be negative.

      blackBG = PlanarModel.build_model(n, dist, tau, isVectorized)
      return blackBG

  #======================== build_spherical_blackBG ========================
  #
  @staticmethod
  def build_spherical_blackBG(dist, tau, isVectorized = True):
      '''!
      @brief  Build a black color-based background model. The assumption is that
              the background colors are in the black corner of the color cube
              and can be split from the foreground colors using a spherical
              cutting surface.  The center is set to be the origin.

      The good news about this kind of model is that it does not matter whether
      the image is RGB (typical) or BGR (OpenCV type).  The corner region containing 
      black is still around (0,0,0).

      @param[in]  dist    Locate of separating boundary. 
      @param[in]  tau     The threshold / margin to apply.
      '''

      c = np.array([[0], [0], [0]])
      blackBG = SphericalModel.build_model(c, dist, tau, isVectorized)
      return blackBG

  #========================== build_model_whiteBG ==========================
  #
  @staticmethod
  def build_model_whiteBG(dist, tau, isVectorized = True):
      '''!
      @brief  Build a white color-based background model. This is rare but
              provided just in case.  The sign is flipped to cut off the
              more extremal white color values.

      The good news about this kind of model is that it does not matter whether
      the image is RGB (typical) or BGR (OpenCV type).  The corner region containing 
      black is still around (0,0,0).

      @param[in]  dist    Locate of separating boundary (should be positive). 
      @param[in]  tau     The threshold / margin to apply.
      '''

      n = np.array(-[1, 1, 1])/np.sqrt(3)      # Points negative to white color along gray line.
                                               # Dist should be positive.

      whiteBG = PlanarModel.build_model(n, dist, tau, isVectorized)
      return whiteBG


  #============================= calibrate_from_data =============================
  #
  #
  @staticmethod
  def calibrate_from_data(bgI, fgI):
   
      pass
      # @todo To be coded up.

  #============================= calibrate_from_image ============================
  #
  #
  @staticmethod
  def calibrate_from_image(I, bgI):
   
      pass
      # @todo To be coded up.

  #================================ load ===============================
  #
  @staticmethod
  def load(fileName):
      fptr = h5py.File(fileName,"r")

      gptr = fptr.get("bgmodel.inCorner")

      bgModel = None
      for name in gptr:
        if   (name == 'PlanarModel'):
            bgModel = PlanarModel.loadFrom(gptr)
        elif (name == 'SphericalModel'):
            bgModel = SphericalModel.loadFrom(gptr)

      theDetector = inCorner()
      if (bgModel is None):
          print("Uh-oh: No background inCorner model found to load.")
      else:
          theDetector.set_model(bgModel)

      return theDetector

  #======================= inCorner/buildFromCfg =======================
  #
  @staticmethod
  def buildFromCfg(theConfig, processor=None):
    '''!
    @brief  Build an inCorner instance from an algorithm configuration instance.

    @param[out] bgDet   Instantiated inCorner background model detector.
    '''

    if (theConfig.cutModel == 'Planar'):
      cutModel = PlanarModel.buildFromCfg(theConfig.cutParmsPlanar, theConfig.tau, 
                                                           theConfig.isVectorized)
      bgDet = inCorner(processor, cutModel)
      return bgDet

    elif (theConfig.cutModel == 'Spherical'):
      #DEBUG
      print('Yup it is Spherical. Being tested. Delete this print if it works.')
      cutModel = SphericalModel.buildFromCfg(theConfig.cutParmsSpherical, theConfig.tau, 
                                                           theConfig.isVectorized)
      bgDet = inCorner(processor, cutModel)
      return bgDet

    else:
      print('Something is off!!')
      print(theConfig)
      return None
  

#
#-----------------------------------------------------------------------------------
#================================ inCornerEstimator ================================
#-----------------------------------------------------------------------------------
#
class inCornerEstimator(inCorner):

  def __init__(self, processor = None, bgMod = None):

    super(inCornerEstimator,self).__init__(processor, bgMod) 

    self.mI           = None
    self.maxMargin    = None

  #============================= estimate_clear ============================
  #
  def estimate_clear(self):

    self.maxMargin = None

  #================================ measure ================================
  #
  def measure(self, I):
    '''!
    @brief  Apply the appearance-based margin calculator to image.

    Computes the margin for the image provided.  If the measurements are to 
    be used for updating the model, then the full process is needed since it
    includes the model adaptation.

    @param[in]  I   Image to test on.
    '''
    super().measure(I)
    self.mI = self.calc_margin(I)


  #================================= adapt =================================
  #
  #
  def adapt(self):

    if (self.maxMargin is None):
        self.maxMargin = self.mI
    else:
        self.maxMargin = np.maximum(self.maxMargin, self.mI)

  #======================== apply_estimated_margins ========================
  #
  def apply_estimated_margins(self):

    self.bgModel.adjustThreshold(self.bgModel.tau + self.maxMargin)


  #========================= calibrateFromRGBStream ========================
  #
  # @brief  Calibrate model using a camera class RGB stream. Return calibrated
  #         model.
  #         
  # The stream is presumed to be a color stream only.  Code is presumed to
  # work but not confirmed.
  #
  # @todo   Confirm and correct as needed.
  #
  def calibrateFromRGBDStream(self, theStream, incVis = False):

    while(True):
      rgb, success = theStream.get_frame()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      self.process(rgb)

      if (incVis):
        bgmask = self.getState()
        display.rgb_binary_cv(rgb, bgmask.x, ratio=0.5, window_name="RGB+Mask")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break

    self.apply_estimated_margins()


  #========================== refineFromRGBDStream =========================
  #
  # @brief  Refine model using a camera class RGBD stream. Return calibrated
  #         model.
  #         
  # The stream is presumed to be a depth + color stream as obtained from
  # a Realsense camera but only the color stream is processed.  Code is 
  # not as generic as it could be.
  #
  # @todo   Modify to be a bit more generic.
  #
  def refineFromRGBDStream(self, theStream, incVis = False):

    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      self.process(rgb)

      if (incVis):
        bgmask = self.getState()
        display.rgb_binary_cv(rgb, bgmask.x, ratio=0.5, window_name="RGB+Mask")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break

  #======================== calibrateFromRGBDStream ========================
  #
  # @brief  Calibrate model using a camera class RGBD stream. Return calibrated
  #         model.
  #         
  # The stream is presumed to be a depth + color stream as obtained from
  # a Realsense camera but only the color stream is processed.  Code is 
  # not as generic as it could be.
  #
  # @todo   Modify to be a bit more generic.
  #
  def calibrateFromRGBDStream(self, theStream, incVis = False):

    self.refineFromRGBDStream(theStream, incVis);
    self.apply_estimated_margins()

  #======================== maskRegionFromRGBDStream =======================
  #
  # @brief  Use model parameters and iamge stream to recover largest background
  #         region in image.  That will generate the mask.
  #
  def maskRegionFromRGBDStream(self, theStream, incVis = False):

    roiIntersect = None

    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      self.process(rgb)

      #-- Below is code that attempts to recover largest background
      #     region.  Performs some logic to remove inconsistent
      #     detections at the pixel level.
      #
      bgmask = self.getState()

      bglabeled = measure.label(bgmask.x, connectivity=2)
      bgprops   = measure.regionprops_table(bglabeled, properties=['label','area'])

      bigl  = np.argmax(bgprops['area'])
      bgroi = (bglabeled == bgprops['label'][bigl])

      if roiIntersect is None:
        roiIntersect = bgroi
      else:
        roiIntersect = np.logical_and(roiIntersect, bgroi)

      if (incVis):
        bgM = cv2.cvtColor(255-roiIntersect.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        fgI = cv2.bitwise_and(rgb, bgM)

        display.rgb_binary_cv(fgI, roiIntersect, ratio=0.25, window_name="Region Mask")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break
      
    return roiIntersect

  #================================== info =================================
  #
  def info(self):
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    gds = fPtr.create_group("bgmodel.inCorner")
    print("Creating group: bgmodel.inCorner")

    self.bgModel.saveTo(gds)

    #DELETE IF NOT NEEDED.
    #configStr = self.config.dump()
    #wsds.create_dataset("configuration", data=configStr)


  ##============================== saveCfg ============================== 
  ##
  #def saveCfG(self, outFile): # Save to YAML file.
  #  '''!
  #  @brief  Save current instance to a configuration file.
  #  '''
  #  with open(outFile,'w') as file:
  #    file.write(self.config.dump())
  #    file.close()


  #
  #---------------------------------------------------------------------
  #====================== Static Member Functions ======================
  #---------------------------------------------------------------------
  #

  #==================== buildAndCalibrateFromConfig ====================
  #
  # @brief  build and calibrate onWorkspace model from an initial config 
  #         and a camera class streaming camera. Return instantiated and 
  #         calibrated model.
  #         
  # The stream is presumed to be a depth + color stream as obtained from
  # a Realsense camera but only the color stream is processed.  Code is 
  # not as generic as it could be.
  #
  # @todo   Modify to be a bit more generic.
  #
  @staticmethod
  def buildAndCalibrateFromConfig(theConfig, theStream, incVis = False):
    return

    bgModel = inCornerEstimator( theConfig )
 
    while(True):
      rgb, dep, success = theStream.get_frames()
      if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

      bgModel.process(rgb)

      if (incVis):
        bgS = bgModel.getState()

        bgIm = cv2.cvtColor(bgS.bgIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        display.rgb_cv(bgIm, ratio=0.25, window_name="RGB")

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
        break

    display.close_cv("RGB")
    return bgModel



#
#============================ detector.fgmodel.inCorner ============================
