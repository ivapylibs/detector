#=========================== image02_depthclip ===========================
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
#=========================== image02_depthclip ===========================

# 
# @file     image02_depthclip.py
#
# @author   Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2021/07/11 [modified]
#
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quitf
#=========================== image02_depthclip ===========================

#==[0] Prep the environment.

import improcessor.basic as improcessor
from detector import inImage
import cv2
import os

#==[1] Create a image sequence that can be thresholded. Define threshold

fpath = os.path.realpath(__file__)
cpath = fpath.rsplit('/', 1)[0]

cap = cv2.VideoCapture(cpath+'/data/depth_proc.avi')

thresh = [45,255]

#==[2] Instantiate inImage detector with a image processor that does
#      the thresholding.

improc = improcessor.basic(cv2.cvtColor, (cv2.COLOR_BGR2GRAY,),\
                           cv2.threshold,(thresh[0],thresh[1],cv2.THRESH_BINARY))

imdet = inImage.image(improc)

#==[3] Apply and visualize.

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    imdet.measure(frame)
    # Display the resulting frame

    cv2.imshow('Demo', imdet.Ip[1])

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#
#=========================== image02_depthclip ===========================
