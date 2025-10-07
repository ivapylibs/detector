from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandOutput:
    label: str
    present: bool
    score: float                   # np.nan if absent
    landmarks: np.ndarray          # (21,3) float32; np.nan if absent
    palm: Optional[np.ndarray] = None    # (6,3) or None
    fingers: Optional[np.ndarray] = None # (15,3) or None
    centroid: Optional[np.ndarray] = None  # (2,) normalized xy or None

def make_hand_nan(label: str) -> HandOutput:
    return HandOutput(
        label=label,
        present=False,
        score=np.nan,
        landmarks=np.full((21,3), np.nan, np.float32),
        palm=None,
        fingers=None,
        centroid=None,
    )

class MediaPipeHandsDetector:
    def __init__(self, max_hands: int = 2, det_conf: float = 0.5, track_conf: float = 0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_complexity=1,
        )
        self._palm_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)

    def close(self):
        self._hands.close()

    def detect(self, frame_bgr) -> List[HandOutput]:
        L, R = make_hand_nan("left"), make_hand_nan("right")

        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not getattr(res, "multi_hand_landmarks", None):
            return [L, R]

        # Collect all detections as candidates
        cands: list[HandOutput] = []
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
            palm    = pts[self._palm_idx, :]
            fingers = np.delete(pts, self._palm_idx, axis=0)
            lab     = handed.classification[0].label.lower()   # "left" or "right"
            score   = float(handed.classification[0].score)
            cands.append(HandOutput(
                label=lab, present=True, score=score,
                landmarks=pts, palm=palm, fingers=fingers,
                centroid=palm[:, :2].mean(axis=0)
            ))

        if not cands:
            return [L, R]

        # Best left and best right by score
        left_cands  = [c for c in cands if c.label == "left"]
        right_cands = [c for c in cands if c.label == "right"]
        left_cands.sort(key=lambda x: x.score, reverse=True)
        right_cands.sort(key=lambda x: x.score, reverse=True)

        if left_cands:
            L = left_cands[0]
        if right_cands:
            R = right_cands[0]

        # If we only got one slot filled but there are 2+ total detections,
        # put the second-best (overall) into the empty slot (relabel for downstream consistency).
        if (L.present + R.present) < 2 and len(cands) >= 2:
            cands_sorted = sorted(cands, key=lambda x: x.score, reverse=True)
            best, second = cands_sorted[0], cands_sorted[1]
            # If one slot empty, assign second there regardless of its original label
            if not L.present:
                L = HandOutput(**{**second.__dict__, "label": "left"})
            elif not R.present:
                R = HandOutput(**{**second.__dict__, "label": "right"})

        return [L, R]
