from __future__ import annotations

import cv2
import numpy as np


def read_image_unicode_safe(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
