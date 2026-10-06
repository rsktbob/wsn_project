"""LinUCB-Ring-SETSpriority：Ring_SETSpriority（C4 PriorityEncoding），每組 searcher-good 用 LinUCB 選區段。"""

from __future__ import annotations

from Algorithm.se.Ring_SETSpriority import Ring_SETSpriority
from Algorithm.se.ring_segment_policy import LinUCBSegmentPolicy


class LinUCB_Ring_SETSpriority(LinUCBSegmentPolicy, Ring_SETSpriority):
    """Ring_SETSpriority with a per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SETSpriority"]
