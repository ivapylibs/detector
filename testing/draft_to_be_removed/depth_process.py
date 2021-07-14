#============================= depth_process =============================
"""

  @brief    Contains preprocess function for the depth frame.  Used to
            process the depth frame videos for the testing

  @author   Yiye Chen          yychen2019@gatech.edu
  @date     2021/07/08

"""
# @quit
#============================= depth_process =============================

import improcessor
import numpy as np
import cv2
import copy

# =========================== to_uint8 =============================
#
# @brief Convert numpy array to uint8 format.
#
def to_uint8(npImg):
    return npImg.asType(np.uint8)

# =========================== save_three_frames =============================
#
# @brief Pick three frames from the sequence.
#
def save_three_frames(depth_frames, name, format):

    # extract 3 frames & save out (raw)
    N, H, W = depth_frames_raw.shape
    frame_id_1 = int(1/4 * N)
    frame_id_2 = int(2/4 * N)
    frame_id_3 = int(3/4 * N)

    for idx, frame_id in enumerate([frame_id_1, frame_id_2, frame_id_3]):
      if format == "npz":
        np.savez(name+"_{}.npz".format(idx),
                 depth_frames=depth_frames_raw[frame_id, :, :])
      elif format == "png":
        cv2.imwrite(name+"_{}.png".format(idx),
                    depth_frames_proc[frame_id, :, :].astype(np.uint8)) 
            

# =========================== main =============================
#
if __name__ == "__main__":
  # Define preprocessing pipeline on depth data for writing.
  #
  preprocess = improcessor.basic(\
                  improcessor.basic.clipTails,(0.05,),\
                  improcessor.basic.scale, (np.array([0, 255]),),\
                  to_uint8, ())

  # Load depth data
  #
  depth_frames_raw = np.load("data/depth_raw.npz")["depth_frames"]
  N, H, W = depth_frames_raw.shape                  # (Num_frames, H, W)

  # For saving the video as uint8.
  #
  video_writer = cv2.VideoWriter("data/depth_proc.avi", 
                                 cv2.VideoWriter_fourcc(*'XVID'),
                                 20.0, (W, H), 0)
  depth_frames_proc = copy.deepcopy(depth_frames_raw).astype(np.uint8)


  # For loop generates a video of uint8 data for image sequence testing.
  #
  for idx in range(depth_frames_raw.shape[0]):

    # pre-process
    #
    depth_frame = preprocess.apply(depth_frames_raw[idx, :, :])
    depth_frames_proc[idx, :, :] = depth_frame

    # save
    #
    video_writer.write(depth_frame)
       
    # show
    #
    cv2.imshow("Display the depth images", depth_frame)    
    if cv2.waitKey(1) == ord('q'):
      break

  # Save 3 frames out of entire sequence for single image testing.
  #
  save_three_frames(depth_frames_proc,\
                    name="data/depth_proc_single",format="png")

#
#============================= depth_process =============================
