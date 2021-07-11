#=========================== image01_threshold ===========================
#
# @brief    Code to test out the simple image detector for a fairly
#           contrived scenario: threshold a grayscale image.
#
# Uses the base inImage class to do the work.  It was originally a
# blank base class that did nothing, but got redefined to have some
# minimal operational ability for simple cases. Also to help with rapid
# prototyping of more complex systems by providing a basic interface to
# build up from.
#
#=========================== image01_threshold ===========================

# 
# @file     image01_threshold.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
#           Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2021/07/03 [created]
#           2021/07/11 [modified]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#=========================== image01_threshold ===========================

#==[0] Prep the environment.

import improcessor.basic as improcessor
from detector import inImage
import cv2
import os

#==[1] Create a simple image that can be thresholded. Define threshold

fpath = os.path.realpath(__file__)
cpath = fpath.rsplit('/', 1)[0]
Image = cv2.imread(cpath+'/data/depth_proc_single_0.png')

thresh = [45,255]

#==[2] Instantiate inImage detector with a image processor that does
#      the thresholding.

improc = improcessor.basic(cv2.cvtColor, (cv2.COLOR_BGR2GRAY,),\
                           cv2.threshold,(thresh[0],thresh[1],cv2.THRESH_BINARY))

imdet = inImage.image(improc)

#==[3] Apply and visualize.

imdet.measure(Image)

if imdet.Ip is not None:
  # Note that the return value from opencv threshold is (ret, Mat)
  cv2.imshow('Demo',imdet.Ip[1])
  cv2.waitKey()
else:
  print('Error found!')

#
#=========================== image01_threshold ===========================
