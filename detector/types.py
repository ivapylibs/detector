# detector/detector/types.py  (new small file)
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class HandOutput:
    label: str                 # "left" | "right"
    present: bool
    score: float               # np.nan if absent
    landmarks: np.ndarray      # (21,3) float32; NaN-filled if absent
    # No palm/fingers/centroid here; trackers will derive them
