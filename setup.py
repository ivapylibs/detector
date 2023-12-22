#!/usr/bin/env/python

"""!
@defgroup   Detector Detector

@brief  A collection of detector implementations for images and signal streams.


Attempts to encapsulate the generic functionality of a detector through this
package.   Here, detector and recognition are loosely identified.  What is
detected can vary: objects, targets, actions, activities, etc.  Basically
anything that can be given semantic meaning from a signal.  Not all semantic
labels need be from detectors though. Some are derivec elsewhere or downstream of
the base detector process.

"""

from setuptools import setup, find_packages

setup(
    name="detector",
    version="1.0.1",
    description="Classes implementing detection based processing pipelines.",
    author="IVALab",
    packages=find_packages(),
    install_requires=[
        "roipoly",
        "numpy",
        "dataclasses",
        "matplotlib",
        "scipy",
        "opencv-contrib-python",
        "improcessor @ git+https://github.com/ivapylibs/improcessor.git",
    ],
)
