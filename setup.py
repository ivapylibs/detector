#!/usr/bin/env/python

"""!
@package detector
    A collection of detector implementations for images and signal streams.
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
