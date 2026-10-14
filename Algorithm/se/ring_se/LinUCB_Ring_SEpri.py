"""LinUCB-Ring-SEpri：Ring_SEpri（PriorityEncoding），每組 searcher-good 用 LinUCB 選區段。

2026-10-13 起和 Ring_SEpri 用同一個 PriorityEncoding。
"""

from __future__ import annotations

from Algorithm.se.ring_se.Ring_SEpri import Ring_SEpri
from Algorithm.se.ring_se.ring_segment_policy import LinUCBSegmentPolicy


class LinUCB_Ring_SEpri(LinUCBSegmentPolicy, Ring_SEpri):
    """Ring_SEpri with a per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SEpri"]
