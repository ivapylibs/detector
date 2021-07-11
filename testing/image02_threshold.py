#!/usr/bin/python3
#=========================== image02_threshold ===========================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image.
#
# Extends ``image01_threshold`` to have a bigger array and visual output of
# the array data and an image.
#
#=========================== image02_threshold ===========================

# 
# @file     image02_threshold.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2021/07/11 [created]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#=========================== image02_threshold ===========================

#==[0] Prep the environment. From most basic to current implementation.
#
import operator
import numpy as np

import cv2

import improcessor.basic as improcessor
import detector.inImage as detector


#==[1] Create a simple image that can be thresholded. Define threshold
#
image = np.zeros((100,100))
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

#==[4] Visualize the output
#
print("Creating window: should see some true values (two adjacent boxes).")
cv2.imshow('Output',255*binDet.Ip.astype(np.uint8))
cv2.waitKey()

#
#=========================== image02_threshold ===========================
