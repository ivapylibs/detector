#=========================== fgmodel/targetNeon ==========================
#
# @class    fgmodel.targetNeon
#
# @brief    General interface for simple appearance-based detection
#           module for images.
#
#
# Neon colors are defined by RGB coordinate values maximizing two of
# three color channels and being of low to mid valued in the third. This
# is defined as less than 180 out of 255, but can be overriden.
#
#=========================== fgmodel/targetNeon ==========================

#
# @file     targetNeon.m
#
# @author   Patricio A. Vela,       pvela@gatech.edu
#           Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2020/08/11 [created]
#           2021/07/10 [modified]
#
# @classf   fgmodel
#=========================== fgmodel/targetNeon ==========================
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import warnings
from appearance import appearance

# Struct for tModel
class TModel(object):
    def __init__(self, mean=None, R=None, T=None, tau=None, classify=None, vectorize=None):
        self.mean =mean
        self.R = R
        self.T = T
        self.tau = tau
        self.classify = classify
        self.vectorize = vectorize

    # @todo Need double check on Ruinian's implementation
    # def classify(self, c):
    #     c_trans = np.abs(np.matmul(self.R, c) + np.repeat(np.expand_dims(self.T, -1), 3, -1))
    #     return np.all(c_trans < np.repeat(np.expand_dims(self.tau, -1), 3, -1), axis=0)

# Struct for mData
class MData(object):
    def __init__(self, pix, data):
        self.pix = pix
        self.data = data

class targetNeon(appearance):

    # ============================= targetNeon ============================
    #
    # @brief  Constructor for neon target FG detector.
    #
    # @param[in]  appMod  The model or parameters for the appearance detector.
    #
    def __init__(self, appMod):
        super(targetNeon, self).__init__(appMod)

    # ============================== measure ==============================
    #
    # @brief  Apply the appearance detector to an image.
    #
    def measure(self, I):

        if self.processor:
            pI = self.processor.apply(I)
        else:
            pI = I

        if self._appMod.vectorize:
            imDat = np.array(pI).reshape(-1,np.pI.shape[2]).T
            fgB = self._appMod.classify(imDat)
            self.fgIm = np.array(fgB).reshape(pI.shape[0], pI.shape[1])
        else:
            self.fgIm = self._appMod.classify(pI)

        # @todo    Eventually shift classification code here. Figure out how to
        #         have different neon models, possibly parametrized/specified
        #         through a structure. For now rolling with struct classifier.

    # ============================ calibSimple ============================
    #
    # @brief  Given a set of data, calibrate the neon target using the
    #         simplest model possible from data that is "neon" colored.
    #
    # Recovers a classification model based on collected data.
    # This method assumes that all data has been collected and organized,
    # and truly represents some kind of "neon" color.  The output variables
    # are structures with the results of the estimation process for the
    # target model. The fields are as follows:
    #
    # tModel (structure)
    #   mean      - mean RGB of user-selected image regions
    #   R         - transformation of color space into new coordinates.
    #   T         - post-transformation centering in new coordinates.
    #   tau       - thresholds to apply to abs. values of transformed colors.
    #   classify  - Classifies matrix color data (needs to be double).
    #   vectorize - Boolean indicating if classification requires vectorization.
    #
    # @param[input]   nData   The neon data as a 3 x N matrix.
    # @param[input]   tCol    Lower value threshold. Optional. Default = 180.
    #
    # @param[output]  tModel  The trained/calibrated target model.
    # @param[mData]   mData   Data used to calibrate the model.
    #
    # =============================== calibSimple ==============================
    def calibSimple(self, nData, tCol=180):
        if not nData:
            raise Exception('Nothing to process.')

        # Get statistical characterization of training pixels.
        tModel= TModel()
        tModel.mean = np.mean(nData,  axis= 1)

        mi = np.argmin(tModel.mean)
        mv = tModel.mean[mi]

        eChan = (tModel.mean > 200).astype(int)

        isNeon = np.sum(eChan) == 2
        isNeon = isNeon & (mv < tCol)

        # Default case is cyan, RGB is (low, high, high).
        #
        tModel.R = np.array([[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], \
                      [1,              0,              0], \
                      [0, -1 / np.sqrt(2), 1 / np.sqrt(2)]]) # Rotated color space.
        tModel.T = -tModel.R @ np.array([mv, 240, 240]).reshape(-1,1) # Shift/center.
        tModel.tau = np.array([20 * np.sqrt(2), 45, 20]).reshape(-1,1)


        # if mi == 0,    % RGB is (low, high, high) = cyan. Already sorted.
        if mi == 1: # RGB is (high, low, high) = magenta.
            tModel.R = np.roll(tModel.R, 1, axis=1)
        elif mi == 2: # RGB is (high, high, low) = yellow (not the best).
            tModel.R = np.roll(tModel.R, 2, axis=1)

        if not isNeon:
            warnings.warn('Color not neon. Should extremize only two of three color channels')
            print('See intermediate processing below. Need two extreme, one not.\n',
                  'Should be reflected in two binary 1 and one binary 0 values.\n')
            print(f'But mean should not have three high values > {tCol}\n\n')

            print(tModel.mean) # @todo Need to double check disp([tModel.mean]'); I do not think we have to transpose the array.
            print(eChan)
            print('Not triggering error. Creating modified model, but it may not work.');
            tModel.T = -tModel.R @ tModel.mean.reshape(3,-1)  # Correct the shift/center.

        tModel.classify = lambda c: all(np.abs(tModel.R @ c + tModel.T) < tModel.tau, 1)
        tModel.vectorize = True

        return tModel

    # ============================ buildSimple ============================
    #
    # @brief  Instantiate a simple neon target model from source data.
    #
    def buildSimple(self, nData, *args):
        if len(args) == 0:
            tModel = self.calibSimple(nData)
        else:
            tModel = self.calibSimple(nData, args)

        neonDet = targetNeon(tModel)

        return neonDet

    # =========================== calibFromImage ==========================
    #
    # @brief    Calibrate the foreground model given an image with target
    #           elements within it.
    #
    # Runs a user-guided foregound selection interface to capture and learn
    # the neon-bright color.  Recover a classification model based on
    # user-specified regions from the image.
    #
    # This method is **user-interactive**; it will display figures and
    # request user selection of image regions.
    #
    # The output variables are structures with the results of the estimation
    # process for the target model. The fields are as follows:
    #
    # mData (structure list)
    #   pix     - pixel locations/values collected for training
    #   data    - source image data at the locations.
    #
    #
    # @param[input]     img     The image to process.
    # @param[input]     nPoly   (Optional) The number of polygons per image
    #                           Default = 1 (if argument missing/empty matrix).
    # @param[input]     fh      Figure to use if given (new figure otherwise).
    #
    # @param[output]    tModel  The trained target model.
    # @param[output]    mData   The training pixel locations and color data.
    #
    def calibFromImage(self, img, nPoly=1, fh=None):
        if fh is None:
            fh = plt.figure()
            newFig = True
        else:
            newFig = False

        fh.title('LEFT-CLICK TO SELECT POLYGON VERTICES THEN RIGHT-CLICK '
                  ' OR DOUBLE CLICK TO SELECT FINAL VERTICE, FOR EACH POLYGON')

        print('\n\n***INSTRUCTIONS***  Use LEFT-CLICK to select a series of polygon '
              'vertices; use RIGHT-CLICK or DOUBLE-CLICKto select the final vertex of each polygon. \n')
        print(f'\t\t\tDo this to define {nPoly} polygons containing pixels you would like ' 
              'to color match.\n\n\n')

        frame_size = img.shape
        img_hdl = plt.figure()
        img_hdl.imshow(np.zeros(frame_size), extent=[0, 1, 0, 1])
        pix = np.array([])
        dat = np.array([])

        # train on each specified frame in image sequence
        ii = 1

        # collect training pixels from each user-defined polygon
        for jj in range(nPoly):
            roi = RoiPoly(color='r')
            b = roi.get_mask(img)
            vals = img.reshape(-1, img.shape[2])
            vals = vals[b.reshape(-1, 1) == 1, :]

            if pix.size == 0:
                pix = vals
            else:
                pix = np.concatenate((pix, vals), axis=0)

            if dat.size == 0:
                dat = vals
            else:
                dat = np.concatenate((dat, vals), axis=0)

        mData = MData(pix.T, dat.T)

        # Get statistical characterization of training pixels.
        #
        tModel = self.calibSimple(mData.data)

        return tModel, mData

    # =========================== buildFromImage ==========================
    #
    # @brief  Instantiate a simple neon target model from image selections
    #
    def buildFromImage(self, img, *args):
        if len(args) == 0:
            tModel = self.calibFromImage(img)
        else:
            tModel = self.calibFromImage(img, args)

        neonDet = targetNeon(tModel)

        return neonDet

    # ========================== calibFromReader ==========================
    #
    # @brief  Calibrate the foreground model given an image sequence
    #         with target elements within it.
    #
    # Runs a user-guided foregound selection interface to capture and learn
    # the neon-bright color. The reader element should have only the frames
    # to process and no more.  Recover a classification model based on
    # user-specified regions from the image frames.
    #
    # This method is **user-interactive**; it will display figures and
    # request user selection of image regions.
    #
    # The output variables are structures with the results of the estimation
    # process for the target model. The fields are as follows:
    #
    # mData (structure list)
    #   pix     - pixel locations/values collected for training
    #   data    - source image data at the locations.
    #
    #
    # @param[input]     imseq   The reader instance with image frames to process.
    # @param[input]     nPoly   (Optional) The number of polygons per image
    #                           Default = 1 (if argument missing/empty matrix).
    # @param[input]     fh      Figure to use if given (new figure otherwise).
    #
    # @param[output]    tModel  The trained target model.
    # @param[output]    mData   The training pixel locations and color data.
    #
    # ========================== targetNeonFromReader =========================
    def calibFromReader(self, imseq, nPoly=1, fh=None):
        if fh is None:
            fh = plt.figure()
            newFig = True
        else:
            newFig = False

        fh.title(num='LEFT-CLICK TO SELECT POLYGON VERTICES THEN RIGHT-CLICK '
                            ' OR DOUBLE CLICK TO SELECT FINAL VERTICE, FOR EACH POLYGON')

        print('\n\n***INSTRUCTIONS***  Use LEFT-CLICK to select a series of polygon '
              'vertices; use RIGHT-CLICK or DOUBLE-CLICKto select the final vertex of each polygon. \n')
        print(f'\t\t\tDo this to define {nPoly} polygons containing pixels you would like ' 
              'to color match.\n\n\n')

        frame_size = imseq.shape
        img_hdl = plt.figure()
        img_hdl.imshow(np.zeros(frame_size), extent=[0, 1, 0, 1])
        pix = np.array([])
        mDatas = []

        ii = 1
        for idx_frame in range(len(imseq)):
            I = imseq[idx_frame]
            img_hdl.title('CData')
            plt.imshow(I.astype(np.uint8))

            for jj in range(nPoly):
                roi = RoiPoly(color='r')
                b = roi.get_mask(I)
                vals = I.reshape(-1, I.shape[2])
                vals = vals[b.reshape(-1, 1) == 1, :]

                if pix.size == 0:
                    pix = vals
                else:
                    pix = np.concatenate((pix, vals), axis=0)

                mDatas.append(MData(b, vals))

            ii += 1

        # Get statistical characterization of training pixels.
        #
        tModel = targetNeon(pix)

        return tModel, mDatas

#
#=========================== fgmodel/targetNeon ==========================

