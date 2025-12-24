# detector/detector/mediapipe_hands.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2
import mediapipe as mp
from perceiver.types import Detections
from skimage.draw import polygon

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
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5, mask_mode: str = "none", mirror: bool = False, use_tips: bool = True, dilate_iter: int = 2, dilate_ks: int = 7):
        self._mp = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            static_image_mode=False,
            model_complexity=1,
        )
        self._palm_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)
        self.mask_mode = mask_mode
        self._mask_mirror = mirror
        self._mask_use_tips = use_tips
        self._mask_dilate_iter = dilate_iter
        self._mask_dilate_ks = dilate_ks
        self._last_mask = None
        self._tip_idx = np.array([4, 8, 12, 16, 20])

        self._prev_hands = []


    def close(self):
        self._mp.close()

    def detect(self, frame_bgr) -> List[HandOutput]:
        """
        Run MediaPipe Hands on either the raw frame or a hand-masked frame
        (if mask_mode is active and we have previous landmarks), then return
        [left, right] HandOutput in stable order. Also updates self._prev_hands
        for next-frame masking.
        """
        # 1) Choose input frame (mask only if configured and we have prior hands)
        frame_in = frame_bgr

        # 2) MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        res = self._mp.process(frame_rgb)

        # 3) Default outputs if no hands
        left, right = make_hand_nan("left"), make_hand_nan("right")
        if not res.multi_hand_landmarks:
            # clear cache so next frame won’t try masking
            self._prev_hands = []
            return [left, right]

        # 4) Collect candidates
        cands: List[HandOutput] = []
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
            palm = pts[self._palm_idx, :]
            fingers = np.delete(pts, self._palm_idx, axis=0)
            lab = handed.classification[0].label.lower()  # "left" / "right"
            score = float(handed.classification[0].score)
            cen = palm[:, :2].mean(axis=0) if np.all(np.isfinite(palm)) else None

            cands.append(HandOutput(lab, True, score, pts, palm, fingers, cen))

        # 5) Pick best two by score and place into (left, right)
        cands.sort(key=lambda h: h.score, reverse=True)
        top = cands[:2]

        if len(top) == 1:
            if top[0].label == "left":
                left = top[0]
            else:
                right = top[0]
        elif len(top) >= 2:
            a, b = top[0], top[1]
            if a.label != b.label:
                left  = a if a.label == "left" else b
                right = b if a.label == "left" else a
            else:
                # same label twice → relabel second for stable (left,right) ordering
                if a.label == "left":
                    left = a
                    right = HandOutput(**{**b.__dict__, "label": "right"})
                else:
                    right = a
                    left  = HandOutput(**{**b.__dict__, "label": "left"})

        # 6) Update previous-hands cache for next-frame masking
        curr_hands = []
        for h in (left, right):
            if (
                getattr(h, "present", False)
                and isinstance(h.landmarks, np.ndarray)
                and h.landmarks.shape == (21, 3)
                and np.all(np.isfinite(h.landmarks))
            ):
                # Store the HandOutput object itself (so getattr works in the masker)
                curr_hands.append(h)
        self._prev_hands = curr_hands
        
        current_mask = None
        if self.mask_mode in ("palm", "hand"):
            current_mask = self._build_polygon_mask(frame_bgr.shape, curr_hands, self.mask_mode)
        self._last_mask = current_mask

        return [left, right]

    

    def _landmarks_to_px(self, pts_norm, w, h, mirror) -> np.ndarray:
        """Convert normalized (N,3) landmarks to pixel (N,2), respecting mirror flag."""
        pts = pts_norm.copy()
        if mirror:
            pts[:, 0] = 1.0 - pts[:, 0]     # flip x in normalized space
        return (pts[:, :2] * np.array([w, h])[None, :]).astype(np.int32)
    

    def _make_hand_mask_from_landmarks(self, frame_shape, hands, mirror=False, use_tips=True, dilate_iter=2, dilate_ks=7) -> np.ndarray:
        """
        Build a binary uint8 mask (H,W) where hand regions are 255.
        - frame_shape: (H, W, 3) or (H, W)
        - hands: list of HandOutput or Tracks with `.present`, `.landmarks`
        - mirror: same logic as your preview overlay
        - use_tips: include fingertips to enlarge hull
        - dilate_iter/ks: grow mask (OK to be bigger than hand)
        """
        if len(frame_shape) == 3:
            H, W = frame_shape[:2]
        else:
            H, W = frame_shape

        mask = np.zeros((H, W), dtype=np.uint8)
        palm_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int32)
        tip_idx  = np.array([4, 8, 12, 16, 20], dtype=np.int32)

        for h in hands:
            if not getattr(h, "present", False):
                continue
            pts = getattr(h, "landmarks", None)
            if not isinstance(pts, np.ndarray) or pts.shape != (21, 3) or not np.all(np.isfinite(pts)):
                continue

            idx = palm_idx
            if use_tips:
                idx = np.concatenate([palm_idx, tip_idx])

            px = self._landmarks_to_px(pts[idx], W, H, mirror)  # (M,2) int
            if px.shape[0] < 3:
                continue

            # Convex hull
            hull = cv2.convexHull(px)
            cv2.fillPoly(mask, [hull], 255)

        if dilate_iter > 0 and dilate_ks > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

        return mask
    
    def _apply_mask(self, frame) -> np.ndarray:
        if self.mask_mode == "hand":
            _use_tips = True
        elif self.mask_mode == "palm":
            _use_tips = False
        else:
            _use_tips = self._mask_use_tips  # default safety

        mask = self._make_hand_mask_from_landmarks(
            frame.shape, self._prev_hands, self._mask_mirror, _use_tips,
            self._mask_dilate_iter, self._mask_dilate_ks
        )

        if np.all(mask == 0):
            self._last_mask = None
            return frame

        self._last_mask = mask
        if frame.ndim == 3:
            mask3 = cv2.merge([mask, mask, mask])
            return cv2.bitwise_and(frame, mask3)
        else:
            return cv2.bitwise_and(frame, frame, mask=mask)
        

    def detect_struct(self, frame_bgr, timestamp: Optional[float] = None) -> Detections:
        """
        Structured detector API:
        - Calls the existing detect() to reuse masking + MediaPipe logic.
        - Returns Detections(items=[...], meta={...}) per Phase-1 API.
        """
        left, right = self.detect(frame_bgr)

        items = []
        
        def _add_if_present(h):
            if (getattr(h, "present", False)
            and isinstance(h.landmarks, np.ndarray)
            and h.landmarks.shape == (21, 3)
            and np.all(np.isfinite(h.landmarks))):
                items.append({
                "label": h.label,                  # "left" / "right"
                "score": float(h.score),
                "landmarks": h.landmarks,          # (21,3) float32, normalized
                "palm": h.palm,                    # (6,3) float32, normalized
                "fingers": h.fingers,              # (15,3) float32, normalized
                "centroid": h.centroid,            # (2,) float32, normalized or None
                })
            
        _add_if_present(right)
        _add_if_present(left)

        H, W = frame_bgr.shape[:2]
        meta = {
            "image_height": H,
            "image_width": W,
            "timestamp": timestamp,
            "source": "mediapipe_hands",
            "mask_mode": self.mask_mode,     # "none" | "palm" | "hand"
            "mirror": self._mask_mirror,
            "image_size": (H, W),
            "num_detected": len(items),
            # (optional) you can stash debug like "used_mask": self._last_mask is not None
        }
        if self.mask_mode != "none" and self._last_mask is not None:
            meta["mask"] = self._last_mask

        return Detections(items=items, meta=meta)
    
    
    def _build_polygon_mask(self, frame_shape, hand_outputs, mask_mode) -> Optional[np.ndarray]:
        if len(frame_shape) == 3:
            H, W = frame_shape[:2]
        else:
            H, W = frame_shape

        mask = np.zeros((H, W), dtype=bool)

        if mask_mode == "palm":
            polygon_idx = self._palm_idx
        elif mask_mode == "hand":
            polygon_idx = np.concatenate([self._palm_idx, self._tip_idx])
        else:
            # No masking requested
            return None

        for h in hand_outputs:
            if (
                getattr(h, "present", False)
                and isinstance(h.landmarks, np.ndarray)
                and h.landmarks.shape == (21, 3)
                and np.all(np.isfinite(h.landmarks))
            ):
              # All landmarks in pixel coords, then select the subset
                pts_norm = h.landmarks                          # (21, 3)
                pts_px_all = self._landmarks_to_px(
                    pts_norm, W, H, mirror=self._mask_mirror
                )                                               # (21, 2)
                pts_px = pts_px_all[polygon_idx]                # (M, 2)

                if pts_px.shape[0] < 3:
                    continue

                # Convex hull to order boundary points
                hull = cv2.convexHull(pts_px)                   # (K, 1, 2) or (K, 2)
                hull = hull.reshape(-1, 2)                      # (K, 2)

                rs = hull[:, 1]  # y / rows
                cs = hull[:, 0]  # x / cols

                rr, cc = polygon(rs, cs, shape=(H, W))
                mask[rr, cc] = True

        if mask.any():
            return mask
        else:
            return None

