# =========================== fgmodel/targetSG ==========================
"""
 @class    fgmodel.targetSG

 @brief    Interfaces for the target detection module 
            based on the single Gaussian RGB color modeling


 The target pixels are assumed to have similar RGB value, and is modeled as a
 single Guassian distribution.  It then transform the
 distribution into a Gaussian with uncorrelated components by
 diagonalizing the covariance matrix. The fg detection is done by
 thresholding each component independently in the transformed color space(tilt):
       |color_tilt_i - mu_tilt_i| < tau * cov_tilt_i, for all i=1,2,3
 tau can be user-input

 The interface is adapted from the fgmodel/targetNeon.
"""
# =========================== fgmodel/targetSG ==========================
"""
 @file     targetSG.m

 @author  Yiye Chen,       yychen2019@gatech.edu
 @date     2021/02/28 [Matlab version]
           2021/07/15 [Python version]

 @classf   fgmodel
"""
# =========================== fgmodel/targetSG ==========================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from roipoly import RoiPoly
from dataclasses import dataclass

from detector.fgmodel.appearance import appearance
from improcessor.basic import basic
from improcessor.mask import mask as maskproc

@dataclass
class Params():
    """
    @param[in]  det_th          Det_th times the eigenvalue will serve as the threshold
    @param[in]  processor       An instance of the improcessor.basic. The preprocessor before the detection
    """
    det_th: int or float = 30 
    processor: basic = None

class tModel_SG():
    """
    The target model class storing the statistics of the target color
    """
    def __init__(self, th=30):
        self.mu = None
        self.sig = None
        self.sigEignVec = None #row vec
        self.sigEignVal = None

        self.has_Eign = False
        self.th = th

    def getSigEign(self):
        self.sigEignVal, self.sigEignVec = np.linalg.eig(self.sig)
        self.has_Eign = True
    
    def classify(self, nData):
        """
        @param[in]  nData       (d, N)
        """
        T = self.mu
        R = self.sigEignVec.T #orthonormal
        nData_trans = R @ (nData - T[:, None])
        mask = np.all(
            np.abs(nData_trans) < self.th * self.sigEignVal[-1, None],
            axis = 0
        )
        return mask 
    

class targetSG(appearance):
    """
    The single-Gaussian based target detection class.
    The target color is modeled as a single-Guassian distribution
    """
    def __init__(self, tModel: tModel_SG=tModel_SG(), params:Params=Params()):
        """The targetSG constructor

        Args:
            tModel (tModel_SG, optional): The target model class storing the single Gaussian model statistics.\
                 Defaults to tModel_SG(), which is an empty model and requires further calibration.
            params (Params, optional): Additional parameters stored as the targetSG.Params instance. Defaults to Params().
        """

        super().__init__(tModel, None)
        self.params = params
        if self.params.det_th is not None:
            self.update_th(self.params.det_th)

    def update_th(self, val):
        self.params.det_th = val
        self._appMod.th = val

    def measure(self, I):
        # TODO: from the way appearance init inImage, it seems that self.processor will never be instantiated?
        if self.params.processor != None:
            pI = self.params.processor.apply(I)
        else:
            pI = I
        pI = np.array(pI)

        if not self._appMod.has_Eign:
            self._appMod.getSigEign()

        # For now only implement the vectorize version
        imDat = pI.reshape(-1, pI.shape[-1]).T
        mask_vec = self._appMod.classify(imDat)
        self.fgIm = mask_vec.reshape(pI.shape[:2])
    
    def calibrate(self, mode="ImgDiff", *args):
        """Calibrate the stored app model

        This permit re-calibrate the stored appearance model, which might be uninitialized or outdated
        Now will simply update 
        """
        # TODO: recalibrate of the model. 
        assert mode in ["Simple", "ImgDiff", "FromImg"]

        tModel = None
        self._appMod = tModel

    def saveMod(self, filename):
        return None
    
    def loadMod(self, filename):
        return None

    @staticmethod
    def _calibSimple(nData):
        """
        calibrate the target model given a set of target pixels 
        @brief  Given a set of data, calibrate the target using the
           Gaussian model from data that is of the same color.

        Recovers a classification model based on collected data. 
        This method assumes that all data has been collected and organized,
        and is drawn from the same target color distribution. The output variables
        are structures with the results of the estimation process for the
        target model. The model class and teh populated fields are:

        tModel (type: tModel_SG):
            self.mu     - The estimated Gaussian mean from the target color samples
            self.sig    - The estimated Gaussian covariance matrix from the target color samples
        
        @param[in]  nData       (3, N)
        @param[out] tModel
        """

        tModel= tModel_SG()
        # Max log-likelihood estimation
        tModel.mu = np.mean(nData, axis=1)
        delta = (nData - tModel.mu.reshape(-1, 1))
        tModel.sig = np.mean(
            delta[:, None, :] * delta[None, :, :],
            axis=2
        )

        return tModel 
    
    @staticmethod
    def buildSimple():
        """
        Return a targetSG detector instance given a set of target pixels
        """
        return None
    
    @staticmethod
    def _calibFromImage(img, nPoly=1, fh=None, *args, **kwargs):
        """
        @brief  Calibrate the foreground model given an image sequence
             with target elements within it.

        Runs a user-guided foregound selection interface to capture and learn
        the target color. The reader element should have only the frames
        to process and no more.  Recover a classification model based on
        user-specified regions from the image frames.

        This method is **user-interactive**; it will display figures and
        request user selection of image regions.

        The output variables are structures with the results of the estimation
        process for the target model. The fields are as follows:
    
        mData (dict)
          pix     - (2, N) pixel locations/values collected for training
          data    - (3, N) source image data at the locations.
   
    
        @param[input]     imseq   The reader instance with image frames to process.
        @param[input]     nPoly   (Optional) The number of polygons per image
                                  Default = 1 (if argument missing/empty matrix).
        @param[input]     fh      Figure to use if given (new figure otherwise).
    
        @param[output]    tModel  The trained target model.
        @param[output]    mData   The training pixel locations and color data.
        """
        if fh is None:
            fh = plt.figure()
            newFig = True
        else:
            newFig = False

        fh.suptitle('LEFT-CLICK TO SELECT POLYGON VERTICES THEN RIGHT-CLICK '
                  ' OR DOUBLE CLICK TO SELECT FINAL VERTICE, FOR EACH POLYGON')
        ax = fh.add_subplot(1,1,1) # assume only one image will be display
        ax.imshow(img)
        plt.show(block=False)

        print('\n\n***INSTRUCTIONS***  Use LEFT-CLICK to select a series of polygon '
              'vertices; use RIGHT-CLICK or DOUBLE-CLICKto select the final vertex of each polygon. \n')
        print(f'\t\t\tDo this to define {nPoly} polygons containing pixels you would like ' 
              'to color match.\n\n\n')

        pix = np.array([])
        dat = np.array([])

        # collect training pixels from each user-defined polygon
        for jj in range(nPoly):
            roi = RoiPoly(color='r', fig=fh)
            b = roi.get_mask(img[:, :, 0])
            vals = img.reshape(-1, img.shape[2])
            vals = vals[b.reshape(-1) == 1, :]

            # TODO: in the Matlab version and targetNeon, the pix are populated in the same way of dat
            # But it should collect the pixel coordinates according to the function comment
            # Do the coordinates for now 
            rowIds, colIds = np.where(b == 1) # this and np.ndarray.reshape are all column-first
            pix_this = np.vstack((rowIds, colIds)) 
            if pix.size == 0:
                pix = pix_this 
            else:
                pix = np.concatenate((pix, pix_this), axis=1)

            if dat.size == 0:
                dat = vals.T
            else:
                dat = np.concatenate((dat, vals.T), axis=1)


        mData = {
            "pix": pix,
            "data": dat 
        }

        tModel = targetSG._calibSimple(dat) 
        return tModel, mData

    @staticmethod 
    def buildFromImage(img, params:Params=Params(), *args, **kwargs):
        """
        @brief  Instantiate a single-Gaussian color target model from image selections

        @param[in]  img                 The image to be calibrated on
        @param[in]  nPoly(optional)     The number of polygon target areas planning to draw. Default is 1
        @param[in]  fh(optional)        The figure handle for displaying the image to draw ROI
        """

        # if kwargs contains nPoly or fh, they will automatically be parsed out
        tModel, mData = targetSG._calibFromImage(img, *args, **kwargs)
        det = targetSG(tModel, params)
        return det 

    @staticmethod
    def _calibImgDiff(bgImg, fgImg, vis=False, ax=None, th=15):
        """Use the image difference to automatically extract the foreground region.
        The extracted foreground region will be used to estimate the target color statistics

        The image difference will requires a background image and an image of the foreground
        object placed on the same background scene.

        Args:
            bgImg (np.ndarray): The background image
            fgImg (np.ndarray): The foreground object image
            vis (bool, optional): If true, will visualize the extracted foreground region
            ax (matplotlib.axes.Axes, optional): The axis for visualizing the extracted foreground region. Defaults to None, which means will create a new figure

        Returns:
            tModel [type]: The calibrated color model
            mData [type]: The target color data used for calibration
        """
        # get the target foreground mask using the image differece
        targetMask = targetSG.get_fg_imgDiff(bgImg, fgImg, th=th)

        # # visualize for debug
        # from camera.utils.display import display_images_cv
        # import cv2
        # targetImg = targetMask[:,:, np.newaxis].astype(np.uint8) * fgImg 
        # display_images_cv([targetImg[:, :, ::-1]], ratio=0.5, window_name="The target color")
        # cv2.waitKey()
        # exit()

        # prepare the data
        vals = fgImg.reshape(-1, fgImg.shape[2])
        vals = vals[targetMask.reshape(-1) == 1, :]
        rowIds, colIds = np.where(targetMask == 1) # this and np.ndarray.reshape are all column-first

        pix = np.vstack((rowIds, colIds)) 
        dat = vals.T

        # calibrate
        mData = {
            "pix": pix,
            "data": dat 
        }
        tModel = targetSG._calibSimple(dat) 

        # visualize
        if vis:
            if ax is None:
                plt.figure()
                ax = plt.gca()
            ax.imshow(fgImg * targetMask[:,:, np.newaxis]) 
            #ax.imshow(targetMask)
            ax.set_title("The target color")

        return tModel, mData            

            
    @staticmethod
    def get_fg_imgDiff(bgImg, fgImg, th):
        """ Use the image difference to get the foreground mask

        Args:
            bgImg (np.ndarray, (H,W,3)): The background image
            fgImg (np.ndarray, (H, W, 3)): The foreground image
            th (int): The threshold for the image difference map

        Returns:
            mask [np.ndarray, (H, W)]: The foreground mask
        """
        imgDiff = np.abs(
            np.mean(bgImg, axis=2) - np.mean(fgImg, axis=2)
        )
        mask = (imgDiff > th)

        # get the largest connected componets and perform erosion
        postproc = maskproc(
            maskproc.getLargestCC, (),
            maskproc.erode, (np.ones((50,50), dtype=bool),)
        )
        mask = postproc.apply(mask)
        
        return mask 

    @staticmethod 
    def buildImgDiff(bgImg, fgImg, vis=False, ax=None, params:Params = Params()):
        """[summary]

        Args:
        
            bgImg (np.ndarray): The background image
            fgImg (np.ndarray): The foreground object image
            vis (bool, optional): If true, will visualize the extracted foreground region
            ax (matplotlib.axes.Axes, optional): The axis for visualizing the extracted foreground region. Defaults to None, which means will create a new figure
            params (Params, optional): [description]. Defaults to Params().

        Returns:
            det [targetSG.targetSG]: A targetSG instance
        """
        tModel, mData = targetSG._calibImgDiff(bgImg, fgImg, vis=vis, ax=ax)
        det = targetSG(tModel, params)
        return det