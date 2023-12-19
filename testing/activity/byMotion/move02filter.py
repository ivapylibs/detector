#!/usr/bin/python
#=============================== move02filter ==============================
'''!
'''
#=============================== move02filter ==============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/08/11              [created]
#
# NOTES:
#   columns 80+
#   indent is 2 spaces
#   tab is 4 spaces with conversion.
#
#=============================== move02filter ==============================

import numpy as np
import camera.utils.display as display
import matplotlib.pyplot as plot
import detector.activity.byMotion as ad
import estimator.dtfilters as df


movcfg = ad.CfgMoving()
movcfg.tau = 0.1
movdet = ad.isMoving(movcfg)

A = np.array([[1.0, 6/50], [0, 1]])
C = np.array([[1.0, 0]])
L = df.calcGainByDARE(A, C, 50*np.identity(2), 0.05)
xfilt  = df.Linear(A, C, L, np.array([[1],[0]]))

tdat = np.linspace(0,6,50)
xdat = np.zeros((2, 50))
zdat = []
ii = 0

#xdat[:,0] = np.array(np.ndarray.flatten(xfilt.x_hat))

for t in tdat:
  xfilt.process(np.cos(t))
  movdet.detect(xfilt.x_hat[1])
  zdat.append(movdet.z.value)
  xdat[:,ii] = np.transpose(np.ndarray.flatten(xfilt.x_hat))
  ii = ii + 1

f = display.plotfig(None, None)

f.num, f.axis = plot.subplots()
ax2 = f.axis.twinx()

f.axis.plot(tdat, np.transpose(xdat))
ax2.plot(tdat, zdat)

print('There is a gain + phase shift in velocity estimate.')
print('Only way to really capture is probably through smoother and')
print('  a delayed estimate. At least delay is known/controlled.')
plot.show()

#
#=============================== move02filter ==============================
