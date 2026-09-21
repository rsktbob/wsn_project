"""定義所有最佳化演算法共用的染色體編碼介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoding(ABC):
    """表示一個尚未解碼的候選解。

    ``code`` 保存染色體；各子類別負責將它解碼成排程與路由 ``State``。
    問題拓樸與能量資料仍由外部的 ``Problem`` 管理，避免編碼物件持有
    容易過期的問題狀態。
    """

    def __init__(self, code=None, *, length=None):
        """建立編碼，並驗證既有染色體或預先指定的長度。"""
        self._code = None
        self.length = None if length is None else int(length)
        if code is not None:
            self._assign_code(code, copy=True)
        elif self.length is None:
            raise ValueError("code must be provided when length is unknown")

    def _assign_code(self, code, *, copy):
        """檢查一維染色體的形狀，並依 ``copy`` 決定是否複製資料。"""
        # 保留呼叫端的 dtype。部分連續型演算法會先保存浮點搜尋位置，
        # 到解碼時才離散化；若在此強制轉成整數，會改變原本搜尋行為。
        values = np.asarray(code)
        if values.ndim != 1:
            raise ValueError("code must be a one-dimensional array")
        if self.length is not None and values.shape != (self.length,):
            raise ValueError(
                f"code must have shape ({self.length},), got {values.shape}"
            )
        self.length = len(values)
        self._code = values.copy() if copy else values

    @property
    def code(self):
        """取得目前的染色體陣列；尚未初始化時回傳 ``None``。"""
        return self._code

    @code.setter
    def code(self, values):
        """設定染色體並重新驗證維度；``None`` 表示清除染色體。"""
        if values is None:
            self._code = None
            return
        self._assign_code(values, copy=False)

    @abstractmethod
    def copy(self):
        """複製編碼，使新物件可獨立修改染色體。"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, problem):
        """依指定 ``problem`` 將染色體解碼成獨立的 ``State``。"""
        raise NotImplementedError


def swap_segment(parent1, parent2, index1, index2):
    """Exchange one code segment and return two Encoding children."""
    if parent1.__class__ is not parent2.__class__:
        raise TypeError("segment-swap parents must use the same encoding")
    code1 = parent1.code
    code2 = parent2.code

    left = np.concatenate(
        (code1[:index1], code2[index1:index2], code1[index2:])
    )
    right = np.concatenate(
        (code2[:index1], code1[index1:index2], code2[index2:])
    )

    encoding_type = parent1.__class__
    return encoding_type(left), encoding_type(right)


__all__ = ["Encoding", "swap_segment"]
