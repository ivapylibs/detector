#!/usr/bin/python
#============================== tgm06depth435 ==============================
"""
@brief  Red glove segmentation refinement from base configuration and
        application to video stream afterwards.

Expands on the ''tgm03glove'' script by using encapsulated calibration or
refinement member functions to shorten code.  The image sequence is obtained
from a (presumed available) Intel Realsense D435 camera, or any other camera
compatible with the D435 camera interface (in principle).

Execution:
----------
Requires user input.

There are two phases of operation.  The first does a little model updating
until the `q` key is pressed.  The second freezes the glove target model and
performs color-based detection.  Hitting `q` again quits the routine.

Assumes availability of an Intel Realsense D435 (or compatible) stream.

"""
#============================== tgm06depth435 ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#============================== tgm06depth435 ==============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.fgmodel.Gaussian as SGM 


#==[0]  Setup the camera and the red glove target model.
#       Use hard coded glove configuration.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('tgm04depth435.yaml')
theStream = d435.D435_Runner(d435_configs)

fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([1100.0,250.0,250.0]) )
fgModel = SGM.fgGaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

#==[1] Run the red glove detector with model updating until `q` pressed.
#       After that test out on stream again as detection only.
#
theStream.start()

fgModel.refineFromRGBDStream(theStream, True)

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)

fgModel.testOnRGBDStream(theStream)

#
#============================== tgm06depth435 ==============================
