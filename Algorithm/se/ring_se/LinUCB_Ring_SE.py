"""LinUCB-Ring-SE：Ring_SE（SensorEncoding），每組 searcher-good 用 LinUCB 選區段。

區段內範圍仍用 Ring_SE 原本的 80/20 規則；細節見 :mod:`ring_segment_policy`。
"""

from __future__ import annotations

from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.ring_se.ring_segment_policy import LinUCBSegmentPolicy


class LinUCB_Ring_SE(LinUCBSegmentPolicy, Ring_SE):
    """Ring_SE with a per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SE"]
