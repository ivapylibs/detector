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
emTable = plt.imread(os.path.join(dPath, "empty_table_big_0.png"))
gloveCali = plt.imread(os.path.join(dPath, "human_puzzle_big_0.png")) 
puzzle = plt.imread(os.path.join(dPath, "puzzle_big_0.png"))


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
        pass

    def get_puzzle_mask(self, img):
        pass

    def get_hand_mask(self, img):
        pass

    def metric(self, gt_mask, pred_mask):
        pass
    
    def visualize(self, img):
        pass


# ======= [3] opencv bg detector
# let's start with a dumb detector and finish the rest part
detector = lambda img: np.ones_like(img, dtype=np.bool)

# ======= [4] test and evaluate
evaluator = BgEva(emTable, gloveCali, puzzle)
test_img = plt.imread(os.path.join(dPath, "human_puzzle_big_1.png"))
pred_mask = detector(test_img)
print(evaluator.evaluate(test_img, pred_mask))


