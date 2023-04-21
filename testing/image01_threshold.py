#!/usr/bin/python
#=========================== image01_threshold ===========================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image.
#
# Uses the base inImage class to do the work.  The inImage class is
# almost a blank base class, except that is does have some minimal
# operational ability for simple cases. Doing so help with rapid
# prototyping of more complex systems by providing a basic interface to
# build up from.
#
#=========================== image01_threshold ===========================

# 
# @file     image01_threshold.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/03 [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#=========================== image01_threshold ===========================

#==[0] Prep the environment. From most basic to current implementation.
#
import operator
import numpy as np
import improcessor.basic as improcessor
import detector.inImage as detector

#==[1] Create a simple image that can be thresholded. Define threshold
#
image = np.zeros((10,25))
image[1:4,4:7]  = 10
image[4:9,7:20] = 10

#==[2] Instantiate inImage detector with an image processor that does
#      the thresholding.
#
improc = improcessor.basic(operator.ge,(7,))
binDet = detector.inImage(improc)

#==[3] Apply and visualize.
#
binDet.process(image)

#==[4] Print the output
#
print("\nShould see some True values: two adjacent boxes.\n")
print(binDet.getState().x.transpose())
print("")

#
#=========================== image01_threshold ===========================
