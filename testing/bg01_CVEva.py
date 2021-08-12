"""
======================================= bgEva ==============================================

        @ brief             The Background detection evaluation routine
                            Test on the opencv background detectors

        @ author            Yiye Chen.          yychen2019@gatech
        @ date              08/07/2021

        Evaluate the detection result by comparing it with the ground 
        truth result.

        The ground truth result is obtained by:
        1. Get puzzle mask: Use image difference.Need an empty table image and a table+puzzle image
        2. Get human mask: Use the human detector.
             The human detector is color single gaussian + Height-based refinement
             So the required data:
                (a) Glove color calibration data - any image with glove in it
                (b) Height estimation data - Empty table

======================================= bgEva ==============================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from Surveillance.layers.human_seg import Human_ColorSG
from Surveillance.layers.human_seg import Params as HParams
from Surveillance.utils.height_estimate import HeightEstimator

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data/large')

# train data. Requires two training images:
emTable = cv2.imread(os.path.join(dPath, "empty_table_big_0.png"))[:, :, ::-1]
emTable_dep = np.load(os.path.join(dPath, "empty_table_big_data_0.npz"))["depth_frame"]
intrinsic = np.load(os.path.join(dPath, "empty_table_big_data_0.npz"))["intrinsics"]

gloveCali = cv2.imread(os.path.join(dPath, "human_puzzle_big_0.png"))[:,:,::-1]
puzzle = cv2.imread(os.path.join(dPath, "puzzle_big_0.png"))[:,:,::-1]


height_estimator = HeightEstimator(intrinsic)
height_estimator.calibrate(emTable_dep)

# ======= [2] define the evaluater
class BgEva():
    def __init__(self, emTable, puzzle, hDetector):

        # GT puzzle
        self.emTable = emTable 
        self.puzzle = puzzle

        # GT human
        self.hDetector = hDetector

        # store middle results
        self.gt_puzzle_mask = None
        self.gt_hand_mask = None
        self.gt_bg_mask = None

    def evaluate(self, img, dep, mask):
        """
        @param[in]  img          The test image
        @param[in]  dep          The test depth map
        @param[in]  mask         The predicted background mask
        @param[out] metric value
        """

        self.get_gt(img, dep)
        return self.metric(self.gt_bg_mask, mask)

    def get_gt(self, img, dep):
        self.gt_puzzle_mask = self.get_puzzle_mask(img)
        self.gt_hand_mask = self.get_hand_mask(img, dep)

        self.gt_bg_mask = np.ones_like(img[:,:,0], dtype=bool)
        self.gt_bg_mask[self.gt_puzzle_mask==1] = 0
        self.gt_bg_mask[self.gt_hand_mask==1] = 0
    
    def get_puzzle_mask(self, img):
        img_diff = np.abs(self.puzzle.astype(np.float) - self.emTable.astype(np.float))
        img_diff = np.mean(img_diff, axis=2) 
        return img_diff > 20.

    def get_hand_mask(self, img, dep):
        postP = lambda init_mask: self.post_process(dep, init_mask)
        self.hDetector.update_params("postprocessor", postP)
        self.hDetector.process(img)
        return self.hDetector.get_mask()

    def metric(self, gt_mask, pred_mask):
        """
        Use IoU for now
        """
        intersection = np.count_nonzero(
            np.logical_and(gt_mask, pred_mask)
        )

        union = np.count_nonzero(
            np.logical_or(gt_mask ,pred_mask)
        )

        return intersection/union
    
    def visualize(self, img, pred_mask):
        """
        visualize those imgs: emTable, puzzle, gt_puzzle_mask, gt_human_mask
        """
        fh, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        axes[0].set_title("The empty table image")
        axes[0].imshow(self.emTable)
        
        axes[1].set_title("The puzzle image")
        axes[1].imshow(self.puzzle)

        axes[2].set_title("The GT puzzle mask")
        axes[2].imshow(self.gt_puzzle_mask, cmap="gray")

        axes[3].set_title("The GT human mask")
        axes[3].imshow(self.gt_hand_mask, cmap="gray")

        axes[4].set_title("The GT table mask")
        axes[4].imshow(self.gt_bg_mask, cmap="gray")

        axes[5].set_title("The predicted mask. \n IoU = {}".format(self.metric(self.gt_bg_mask, pred_mask)))
        axes[5].imshow(pred_mask, cmap="Greys")

    def post_process(self, depth, init_mask):
        """
        The function to:
        (a) get the height map from the depth map
        (b) perform thresholding on the height map and find the connected component to the largest CC of the init_mask
        (c) assuming the hand is reaching out from the top of the image frame, remove all pixels so far below the init_mask as outlier
        """

        # threshold
        height_map = np.abs(height_estimator.apply(depth))

        # get max CC of the init_mask
        labels_init_mask, num_init_labels = ndi.label(init_mask)
        sums = ndi.sum(init_mask, labels_init_mask, np.arange(num_init_labels + 1))
        connected_to_max_init = sums == max(sums)   # by take only the max, the non-largest connected component of the init_mask will be ignored
        init_mask_max_CC = connected_to_max_init[labels_init_mask]

        init_height = height_map[init_mask_max_CC]
        low = np.amin(init_height)
        print("The lowest detected height: {}. ".format(low))
        mask = height_map > low 

        # Connected components of mask 
        labels_mask, num_labels = ndi.label(mask)
        # Check which connected components contain pixels from mask_high.
        sums = ndi.sum(init_mask, labels_mask, np.arange(num_labels + 1))
        connected_to_max_init = sums == max(sums)   # by take only the max, the non-largest connected component of the init_mask will be ignored
        max_connect_mask = connected_to_max_init[labels_mask]

        # remove pixels so far below the init mask
        cols_init = np.where(init_mask==1)[0]
        col_max = np.amax(cols_init)
        final_mask = copy.deepcopy(max_connect_mask)
        final_mask[col_max+10:] = 0

        #plt.subplot(141)
        #plt.imshow(init_mask_max_CC, cmap="gray")
        #plt.subplot(142)
        #plt.imshow(height_map)
        #plt.subplot(143)
        #plt.imshow(mask, cmap="gray")
        #plt.subplot(144)
        #plt.imshow(final_mask, cmap="gray")

        return final_mask



# ===== [3] Construct the evaluator instance
hDetector = Human_ColorSG.buildFromImage(gloveCali, None, None)
evaluator = BgEva(emTable, puzzle, hDetector)

# ======= [3] opencv bg detector
# let's start with a dumb detector and finish the rest part
detector = lambda img: np.ones_like(img[:,:,0], dtype=bool)

# ======= [4] test and evaluate
test_img = cv2.imread(os.path.join(dPath, "human_puzzle_big_2.png"))[:,:,::-1]
test_dep = np.load(os.path.join(dPath, "human_puzzle_big_data_2.npz"))["depth_frame"]

# get predict mask
pred_mask = detector(test_img)
print(evaluator.evaluate(test_img, test_dep, pred_mask))
evaluator.visualize(test_img, pred_mask)

plt.show()
