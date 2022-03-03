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


## installation instruction


```
git clone git@github.com:ivapylibs/detector.git
pip3 install -e detector/
```

The test files are shell command line executable and should work when
invoked, presuming that pip installation has been performed.  If no
modifications to the source code will be performed then the ``-e`` flag
is not neessary (e.g., use the flag if the underlying code will be
modified).



## Dependencies

Requires the installation of the following python packages:

- ```numpy```
- ```opencv-contrib-python```
- ```matplotlib```
- ```roipoly```

