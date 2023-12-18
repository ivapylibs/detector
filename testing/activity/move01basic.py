#!/usr/bin/python
#=============================== move01basic ===============================
'''!
'''
#=============================== move01basic ===============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/08/11              [created]
#
# NOTES:
#   columns 80+
#   indent is 2 spaces
#   tab is 4 spaces with conversion.
#
#=============================== move01basic ===============================

import numpy as np
import detector.activity.simple as ad


movcfg = ad.CfgMoving()
movcfg.tau = 0.2
movdet = ad.isMoving(movcfg)


for t in np.linspace(0,6,50):
  x = np.cos(t)
  movdet.detect(x)
  print((x, movdet.z))



#
#=============================== move01basic ===============================
