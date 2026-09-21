"""CodingState 版本的 GI-GOMEA。"""

from Algorithm.gomea.BaseGIGOMEA import BaseGIGOMEA


class GI_GOMEA(BaseGIGOMEA):
    """使用 SensorEncoding，同時求解 WSN 感測排程與路由的 GI-GOMEA。

    所有行為均由 BaseGIGOMEA 提供；保留此類別作為演算法註冊與選用入口。
    """


GIGOMEA = GI_GOMEA
