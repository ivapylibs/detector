#!/usr/bin/python
#=============================== move03filter ==============================
'''!
@brief  Repeat of move03filter but using pixel level measurements.

Also adjusted to measure at 1Hz, not because that is the rate but due
to simplicity for now since the actual rate is not known.
'''
#=============================== move03filter ==============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/08/11              [created]
#
# NOTES:
#   columns 80+
#   indent is 2 spaces
#   tab is 4 spaces with conversion.
#
#=============================== move03filter ==============================

import numpy as np
import camera.utils.display as display
import matplotlib.pyplot as plot
import detector.activity.byMotion as ad
import estimator.dtfilters as df


movcfg = ad.CfgMoving()
movcfg.tau = 5
movdet = ad.isMovingInImage(movcfg)

A = np.array([[1.0, 0.5], [0, 1.0]])
C = np.array([[1.0, 0]])
L = df.calcGainByDARE(A, C, np.diag([10, 10]), 5)
xfilt  = df.Linear(A, C, L, np.array([[100],[0]]))

nsamp = 200
tdat = np.linspace(0,100,nsamp)
xdat = np.zeros((2, nsamp))
zdat = []
ii = 0

for t in tdat:
  xfilt.process(50 + 50*np.cos(t/5))
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
#=============================== move03filter ==============================
