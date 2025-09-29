# detector/detector/mediapipe_hands.py
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandOutput:
    label: str                     # "left" or "right"
    present: bool
    score: float                   # np.nan if absent
    landmarks: np.ndarray          # (21,3) float32; filled with np.nan if absent
    palm: Optional[np.ndarray] = None   # (6,3) or None
    fingers: Optional[np.ndarray] = None # (15,3) or None

def make_hand_nan(label: str) -> HandOutput:
    return HandOutput(label, False, np.nan,
                      np.full((21,3), np.nan, np.float32),
                      None, None)

class MediaPipeHandsDetector:
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5):
        self._mp = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self._palm_idx = np.array([0,1,5,9,13,17], dtype=np.int64)

    def detect(self, frame_bgr) -> List[HandOutput]:
        left, right = make_hand_nan("left"), make_hand_nan("right")
        res = self._mp.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return [left, right]
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)
            palm = pts[self._palm_idx, :]
            fingers = np.delete(pts, self._palm_idx, axis=0)
            score = handed.classification[0].score
            lab = handed.classification[0].label.lower()
            tgt = left if lab == "left" else right
            if not np.isfinite(tgt.score) or score > tgt.score:
                tgt.present, tgt.score = True, score
                tgt.landmarks, tgt.palm, tgt.fingers = pts, palm, fingers
        return [left, right]

    def close(self): self._mp.close()
