import numpy as np

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


class targetNeon(appearance):
    def __init__(self, appMod):
        super(targetNeon, self).__init__(appMod)

    def measure(self, I):
        if isinstance(self.preprocessor, np.array):
            if self.preprocessor.size != 0:
                pI = np.matmul(I, self.preprocessor) # ToDo: confirm if it is the same
            else:
                pI = I
        else:
            raise Exception('wrong type for preprocessor')
        if self._appMod.vectorize:
            imDat = np.reshape(pI, (1, -1))
            fgB = self.appMod.classify(imDat)
            self.fgIm = np.reshape(fgB, (pI.shape[0], pI.shape[1])) # ToDo: confirm if it is the same
        else:
            self.fgIm = self._appMod.classify(pI)

    def calibSimple(self, nData, tCol=180): # ToDo: mData is not claimed in original Matlab code
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