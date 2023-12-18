

### Single Gaussian depth image modeling. ###
| Filename | Implementation information |
| :------- | :------------------------- |
| sgm01basic     |  Synthetic data in matrix to test/show basic functionality. | 
| sgm02depth435  |  Snags streamed data from RS435 with loaded settings. Learning + deploy phases. |
| | |


### Planar onWorkspace background modeling. ###

| Filename | Implementation information |
| :------- | :------------------------- |
| pws02depth435  |  Planar workspace approach (one-sided onWorkspace Gaussian model). |
| pws03saveload  |  Implements save after learning + load to deploy. |
| pws04calibrate |  Tests static method for build and calibrate using camera class API. |
| pws05saveload  |  Combines pws04 and pws03. Shorter code that pws03. |

### Black inCorner background modeling. ###
| Filename | Implementation information |
| :------- | :------------------------- |
| blackbg01_planar    | Run inCorner background model detector on image, with a black corner model.
| blackbg02_spherical | Run inCorner background model detector on image, with a \
                        black spherical model. |
| blackbg03_spherical | Run inCorner background model detector on image, with a \
                        black spherical model. |
| blackbg04_realsense | Simple code with original Realsense API to snag color imagery \
                        and apply the planar black background background estimation model.\
                        Visualize in window. |
| blackbg05_adjust    | Use Realsense camera API to snag color imagery and apply the planar \
                        black background background estimation model with option to adjust \
                        threshold. |
| blackbg06_margins   | Use Realsense API to collect background classifier margin values across \
                        imagery over time.  Stores the maximum value read while collecting data.  |
| blackbg07_estimate  | Testing out the inCornerEstimator class. |
| blackbg08_saveload  | Testing out save/load of running configuration (after learning phase). |
| blackbg09_calibrate | Creating member function for calibration from camera stream (RGBD). |
