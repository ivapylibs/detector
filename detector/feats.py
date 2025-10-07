import numpy as np
_PALM_IDX = np.array([0,1,5,9,13,17], dtype=np.int64)

def palm_from_landmarks(pts: np.ndarray) -> np.ndarray:
    # pts: (21,3)
    return pts[_PALM_IDX, :]

def fingers_from_landmarks(pts: np.ndarray) -> np.ndarray:
    return np.delete(pts, _PALM_IDX, axis=0)

def centroid_from_palm(palm: np.ndarray) -> np.ndarray:
    # normalized (x,y); palm: (6,3)
    return palm[:, :2].mean(axis=0)
