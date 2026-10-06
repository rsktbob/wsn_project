"""LinUCB-Ring-SETS：Ring_SETS（SensorEncoding），每組 searcher-good 用 LinUCB 選區段。

區段內範圍仍用 Ring_SETS 原本的 80/20 規則；細節見 :mod:`ring_segment_policy`。
"""

from __future__ import annotations

from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.ring_segment_policy import LinUCBSegmentPolicy


class LinUCB_Ring_SETS(LinUCBSegmentPolicy, Ring_SETS):
    """Ring_SETS with a per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SETS"]
