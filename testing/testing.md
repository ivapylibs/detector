
### OpenCV Implementation Tests ###
bg01_Eva            Background detection evaluation routine for opencv background detectors.
bg02_cvGMM:         Experiment with the Opencv GMM-based background substraction methods.
bg03_cvGMM:         Experiment with the Opencv GMM-based background substraction methods.

### Detector API Test Routines ###
image01_threshold       Simple in image detector based on improcessor using hard coded data.
image02_threshold       Simple in image detector based on improcessor with visual output.
image03_inRange         In image detector with improcessor using OpenCV inRange on fake image.
image04_inRange         inImage detector using OpenCV inRange with stored depth data.
image05_inRange         Extend to grab from a depth image sequence (loaded).
image06_inRange         Extend to grab from a depth image sequence (loaded).
image07_targetSG        Test single-Gaussian-color-modeling-based foreground detector.
image08_targetSGf       Test single-Gaussian-color-modeling-based foreground detector with
                        foreground statistics obtained from image differencing.
image09_targetMagenta   Test the Magenta foreground target detector. 

### Black inCorner background modeling. ###
blackbg01_planar    Run inCorner background model detector on image, with a black corner model.
blackbg02_spherical Run inCorner background model detector on image, with a black spherical model.
blackbg03_spherical Run inCorner background model detector on image, with a black spherical model.
blackbg04_realsense Simple code with original Realsense API to snag color imagery and apply the
                    planar black background background estimation model. Visualize in window.
blackbg05_adjust    Use Realsense camera API to snag color imagery and apply the planar
                    black background background estimation model with option to adjust threshold.
blackbg06_margins   Use Realsense API to collect background classifier margin values across
                    imagery over time.  Stores the maximum value read while collecting data.
region01_realsense  Background model processed for largest connected region evaluated over time
