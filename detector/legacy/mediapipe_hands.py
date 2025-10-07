# detector/detector/mediapipe_hands.py
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandOutput:
    label: str                     # "left" or "right"
    present: bool
    score: float                   # np.nan if absent
    landmarks: np.ndarray          # (21,3) float32; NaNs if absent
    palm: Optional[np.ndarray] = None    # (6,3) or None
    fingers: Optional[np.ndarray] = None # (15,3) or None
    centroid: Optional[np.ndarray] = None  # (2,) normalized xy or None

def make_hand_nan(label: str) -> HandOutput:
    return HandOutput(
        label=label,
        present=False,
        score=np.nan,
        landmarks=np.full((21, 3), np.nan, np.float32),
        palm=np.full((6, 3), np.nan, np.float32),
        fingers=np.full((15, 3), np.nan, np.float32),
        centroid=None,
    )

class MediaPipeHandsDetector:
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5):
        self._mp = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            static_image_mode=False,
            model_complexity=1,
        )
        self._palm_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)

    def close(self):
        self._mp.close()

    def detect(self, frame_bgr) -> List[HandOutput]:
        left, right = make_hand_nan("left"), make_hand_nan("right")
        res = self._mp.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return [left, right]

        # collect per-hand candidates
        cands: list[HandOutput] = []
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)  # (21,3)
            palm = pts[self._palm_idx, :]
            fingers = np.delete(pts, self._palm_idx, axis=0)
            lab = handed.classification[0].label.lower()     # "left" / "right"
            score = float(handed.classification[0].score)
            # compute centroid from palm if finite, else None
            cen = palm[:, :2].mean(axis=0) if np.all(np.isfinite(palm)) else None

            cands.append(HandOutput(lab, True, score, pts, palm, fingers, cen))

        # choose best two by score and place into (left, right) slots
        cands.sort(key=lambda h: h.score, reverse=True)
        top = cands[:2]

        if len(top) == 1:
            if top[0].label == "left":
                left = top[0]
            else:
                right = top[0]
            return [left, right]

        a, b = top[0], top[1]
        if a.label != b.label:
            left  = a if a.label == "left" else b
            right = b if a.label == "left" else a
        else:
            # same label twice → relabel second for stable (left,right) output
            if a.label == "left":
                left = a
                right = HandOutput(**{**b.__dict__, "label": "right"})
            else:
                right = a
                left  = HandOutput(**{**b.__dict__, "label": "left"})

        return [left, right]
