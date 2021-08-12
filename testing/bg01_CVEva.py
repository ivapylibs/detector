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
import cv2
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data/large')

# train data. Requires two training images:
emTable = cv2.imread(os.path.join(dPath, "empty_table_big_0.png"))[:, :, ::-1]
gloveCali = cv2.imread(os.path.join(dPath, "human_puzzle_big_0.png"))[:,:,::-1]
puzzle = cv2.imread(os.path.join(dPath, "puzzle_big_0.png"))[:,:,::-1]


# ======= [2] build the evaluater
class BgEva():
    def __init__(self, emTable, gloveCali, puzzle):
        self.emTable = emTable 
        self.gloveCali = gloveCali
        self.puzzle = puzzle

        # store middle results
        self.gt_puzzle_mask = None
        self.gt_hand_mask = None
        self.gt_bg_mask = None

    def evaluate(self, img, mask):
        """
        @param[in]  img          The test image
        @param[in]  mask         The predicted background mask
        @param[out] metric value
        """

        self.get_gt(img)
        return self.metric(self.gt_bg_mask, mask)

    def get_gt(self, img):
        self.gt_puzzle_mask = self.get_puzzle_mask(img)
        self.gt_hand_mask = self.get_hand_mask(img)

        self.gt_bg_mask = np.ones_like(img, dtype=np.bool)
        self.gt_bg_mask[self.gt_puzzle_mask==1] = 0
        self.gt_bg_mask[self.gt_hand_mask==1] = 0
    
    def get_puzzle_mask(self, img):
        img_diff = np.abs(self.puzzle.astype(np.float) - self.emTable.astype(np.float))
        img_diff = np.mean(img_diff, axis=2) 
        return img_diff > 20.

    def get_hand_mask(self, img):
        pass

    def metric(self, gt_mask, pred_mask):
        pass
    
    def visualize(self, img):
        """
        visualize those imgs: emTable, puzzle, gt_puzzle_mask,
        """
        fh, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].set_title("The empty table image")
        axes[0].imshow(self.emTable)
        
        axes[1].set_title("The puzzle image")
        axes[1].imshow(self.puzzle)

        axes[2].set_title("The puzzle mask")
        axes[2].imshow(self.gt_puzzle_mask, cmap="gray")

        pass


# ======= [3] opencv bg detector
# let's start with a dumb detector and finish the rest part
detector = lambda img: np.ones_like(img, dtype=np.bool)

# ======= [4] test and evaluate
evaluator = BgEva(emTable, gloveCali, puzzle)
test_img = plt.imread(os.path.join(dPath, "human_puzzle_big_1.png"))
pred_mask = detector(test_img)
print(evaluator.evaluate(test_img, pred_mask))
evaluator.visualize(test_img)

plt.show()


