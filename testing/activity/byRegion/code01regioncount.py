

import numpy as np





tI = np.zeros([50, 50, 3], dtype='bool')

tI[0:6,0:6,0] = True
tI[10:26,10:26,1] = True
tI[30:46,30:36,2] = True

zCount1 = np.count_nonzero(tI, axis = 0)
print(np.shape(zCount1))
zCount2 = np.sum(zCount1, axis = 0)
print(np.shape(zCount2))
print(zCount2)

print(np.array([ 6*6, 16*16, 16*6]))


