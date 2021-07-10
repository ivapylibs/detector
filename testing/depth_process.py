"""

Contains preprocess function for the depth frame. 
Used to process the depth frame videos for the testing

@author: Yiye Chen          yychen2019@gatech.edu
@date: 07/08/2021

TODO:(Due to my limited experience working with the depth info, below needs discussion)
    1. Consider moving the preprocess function to the camera node if such preprocess is necessary for all down stream tasks
    2. Preprocess function needs improvement? 

"""


import numpy as np
import cv2

def clip(depth_frame):
    # clip. remove the smallest and the largest 5% value
    N = depth_frame.flatten().size
    sorted_values = np.sort(depth_frame.flatten()) 
    th_low = sorted_values[int(N*.05)]
    th_high = sorted_values[int(N*.95)]
    depth_frame[depth_frame < th_low] = th_low
    depth_frame[depth_frame > th_high] = th_high
    return depth_frame

def scale(depth_frame):
    # scale to 0 - 255
    min_val = depth_frame.min() 
    max_val = depth_frame.max()
    depth_frame = (depth_frame - min_val) / max_val * 255
    return depth_frame.astype(np.uint8)

def preprocess(depth_frame):
    """
    preprocess a single frame
    """
    depth_frame = clip(depth_frame) 
    depth_frame = scale(depth_frame)
    return depth_frame

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
            


if __name__ == "__main__":
    # load depth data
    depth_frames_raw = np.load("depth_raw.npz")["depth_frames"] # (Num_frames, H, W)
    N, H, W = depth_frames_raw.shape

    # preprocess and show and save
    video_writer = cv2.VideoWriter(
        "depth_proc.avi", 
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0, (W, H), 0
    )
    import copy
    depth_frames_proc = copy.deepcopy(depth_frames_raw).astype(np.uint8)
    for idx in range(depth_frames_raw.shape[0]):
        # preprocess
        depth_frame = preprocess(depth_frames_raw[idx, :, :])
        depth_frames_proc[idx, :, :] = depth_frame
        # save
        video_writer.write(depth_frame)
        # show
        cv2.imshow("Display the depth images", depth_frame)    
        if cv2.waitKey(1) == ord('q'):
            break


    # save 3 frames out
    save_three_frames(depth_frames_proc, name="depth_proc_single", format="png")


