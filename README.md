# detection
Collection of detection methods. At the moment, it is mostly code
translated over from IVALab Matlab libraries.

Some of the detection methods are built on basic image processing
functions for which there should be OpenCV implementations. In that
case, the OpenCV functionality is encapsulated by classes here that
create a more generic interface. In some cases, the code will be
rewritten for ease of translating the Matlab code. How the translation
functions really depends on what the needs are and how well python
replicates them.

## Install

Install the following repositories from the source:

- [improcessor](https://github.com/ivapylibs/improcessor)

```
git clone git@github.com:ivapylibs/detector.git
pip3 install -e detector/
```

The test files are shell command line executable and should work when
invoked, presuming that pip installation has been performed.  If no
modifications to the source code will be performed then the ``-e`` flag
is not necessary (e.g., use the flag if the underlying code will be
modified).

## Run the unit tests

Please first download the presaved data into the ```REPO_ROOT/testing/data``` from the Dropbox [link](https://www.dropbox.com/sh/l5o65khrzt4amrp/AAD9f-VshrQ8s7XNAWTUSz__a?dl=0)

## Dependencies

Requires the installation of the following python packages:

- ```numpy```
- ```opencv-contrib-python```
- ```matplotlib```
- ```dataclasses```
- ```scipy```
- ```roipoly```
