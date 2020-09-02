import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly

from appearance import appearance

class tModel(object):
    def __init__(self, mean, R, T, tau, vectorize):
        self.mean =mean
        self.R = R
        self.T = T
        self.tau = tau
        self.vectorize =vectorize

    def classify(self, c):
        c_trans = np.abs(np.matmul(self.R, c) + np.repeat(np.expand_dims(self.T, -1), 3, -1))
        return np.all(c_trans < np.repeat(np.expand_dims(self.tau, -1), 3, -1), axis=0)

class mData(object):
    def __init__(self, pix, data):
        self.pix = pix
        self.data = data

def overhead_calib_targetNeon(nData, tCol=180):
    if isinstance(nData, np.array):
        if self.preprocessor.size == 0:
            raise Exception('Nothing to process.')
    else:
        raise Exception('wrong type for preprocessor')

    mean = np.mean(nData, -1)

    mi = np.argmin(mean)
    mv = mean[mi]

    eChan = mean > 200
    eChan = eChan.astype(int)

    isNeon = np.sum(eChan) == 2
    isNeon = isNeon & (mv < tCol)

    R = np.array([[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], \
                  [1, 0, 0], \
                  [0, -1 / np.sqrt(2), 1 / np.sqrt(2)]])
    T = np.matmul(-R, np.array([mv, 240, 240]).T)
    tau = np.array([20 * np.sqrt(2), 45, 20]).T

    if isNeon:
        if mi == 2:
            R = R.transpose(2, 0, 1)
        elif mi == 3:
            R = R.transpose(1, 2, 0)
    else:
        raise Exception('Color not neon. Should extremize only two of three color channels')

    vectorize = True

    tModel = tModel(mean, R, T, tau, vectorize)

    return tModel

class targetNeon(appearance):
    def __init__(self, appMod):
        super(targetNeon, self).__init__(appMod)

    def measure(self, I):
        if isinstance(self.preprocessor, np.array):
            if self.preprocessor.size != 0:
                pI = np.matmul(I, self.preprocessor)
            else:
                pI = I
        else:
            raise Exception('wrong type for preprocessor')

        if self._appMod.vectorize:
            imDat = np.reshape(pI, (1, -1))
            fgB = self.appMod.classify(imDat)
            self.fgIm = np.reshape(fgB, (pI.shape[0], pI.shape[1]))
        else:
            self.fgIm = self._appMod.classify(pI)

    def calibSimple(self, nData, tCol=180):
        if isinstance(nData, np.array):
            if self.preprocessor.size == 0:
                raise Exception('Nothing to process.')
        else:
            raise Exception('wrong type for preprocessor')

        mean = np.mean(nData, -1)

        mi = np.argmin(mean)
        mv = mean[mi]

        eChan = mean > 200
        eChan = eChan.astype(int)

        isNeon = np.sum(eChan) == 2
        isNeon = isNeon & (mv < tCol)

        R = np.array([[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], \
                      [1,              0,              0], \
                      [0, -1 / np.sqrt(2), 1 / np.sqrt(2)]])
        T = np.matmul(-R, np.array([mv, 240, 240]).T)
        tau = np.array([20 * np.sqrt(2), 45, 20]).T

        if isNeon:
            if mi == 2:
                R = R.transpose(2, 0, 1)
            elif mi == 3:
                R = R.transpose(1, 2, 0)
        else:
            raise Exception('Color not neon. Should extremize only two of three color channels')

        vectorize = True

        tModel = tModel(mean, R, T, tau, vectorize)

        return tModel

    def buildSimple(self, nData, *args):
        if len(args) == 0:
            tModel = self.calibSimple(nData)
        else:
            tModel = self.calibSimple(nData, args[:])

        neonDet = targetNeon(tModel)

        return neonDet

    def calibFromImage(self, img, nPoly=1, fh=None):
        if fh is None:
            fh = plt.figure()
            newFig = True
        else:
            newFig = False

        plt.title('LEFT-CLICK TO SELECT POLYGON VERTICES THEN RIGHT-CLICK '
                  ' OR DOUBLE CLICK TO SELECT FINAL VERTICE, FOR EACH POLYGON')

        print('\n\n***INSTRUCTIONS***  Use LEFT-CLICK to select a series of polygon '
              'vertices; use RIGHT-CLICK or DOUBLE-CLICKto select the final vertex of each polygon. \n')
        print('\t\t\tDo this to define {} polygons containing pixels you would like ' 
              'to color match.\n\n\n'.fornat(nPoly))

        frame_size = img.shape
        plt.imshow(img, extent=[0, 1, 0, 1])
        pix = np.array([])
        dat = np.array([])

        # train on each specified frame in image sequence
        ii = 1

        # collect training pixels from each user-defined polygon
        for jj in range(1, nPoly):
            roi = RoiPoly(color='r', fig=fig)
            b = roi.get_mask(img)
            vals = np.reshape(img, (-1, img.shape[-1]))
            vals = vals[np.reshape(b, (-1, 1)) == 1, :]

            if pix.size == 0:
                pix = vals
            else:
                pix = np.concatenate((pix, vals), axis=0)
            if dat.size == 0:
                dat = vals
            else:
                dat = np.concatenate((dat, vals), axis=0)

        mData = mData(pix.transpose(1, 0), dat.transpose(1, 0))

        tModel = overhead_calib_targetNeon(mData.data)

        return tModel, mData

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
        print('\t\t\tDo this to define {} polygons containing pixels you would like ' 
              'to color match.\n\n\n'.fornat(nPoly))

        frame_size = img.shape
        img_hdl = plt.figure()
        img_hdl.imshow(np.zeros(frame_size), extent=[0, 1, 0, 1])
        pix = np.array([])
        mDatas = []

        ii = 1
        for idx_frame in range(len(imseq)):
            I = imseq[idx_frame]
            img_hdl.title('CData')
            plt.imshow(I.astype(np.uint8))

            for jj in range(0, a_num_polys):
                roi = RoiPoly(color='r', fig=fig)
                b = roi.get_mask(img)
                vals = np.reshape(img, (-1, img.shape[-1]))
                vals = vals[np.reshape(b, (-1, 1)) == 1, :]

                if pix.size == 0:
                    pix = vals
                else:
                    pix = np.concatenate((pix, vals), axis=0)

                mDatas.append(mData(b, vals))

            ii += 1

        tModel = targetNeon(pix)

        return tModel, mDatas


