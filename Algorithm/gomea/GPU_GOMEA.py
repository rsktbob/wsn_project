"""以 CUDA 加速 linkage learning 的 WSN GI-GOMEA。"""

from Algorithm.gomea.BaseGIGOMEA import BaseGIGOMEA


class GPU_GOMEA(BaseGIGOMEA):
    """保留 GI-GOM 搜尋語意，使用 CUDA 平行計算每代的 NMI 矩陣。

    目前 GPU 範圍刻意限制在 ``linkage_nmi``。候選解解碼、精確路由、
    fitness 與依序接受交換仍在 CPU 執行，避免把硬體加速誤當成另一套
    搜尋演算法。CUDA 不存在時預設退回 CPU；實驗環境若要避免誤跑 CPU，
    可設定 ``require_cuda=True``。
    """

    def __init__(
        self,
        P,
        population_size=24,
        elite_fraction=0.5,
        statistical_weight=0.7,
        graph_weight=0.3,
        max_linkage_size=16,
        max_linkage_sets=160,
        seed=None,
        cuda_device=0,
        require_cuda=False,
    ):
        """建立預設選用 CUDA 的 SensorEncoding GI-GOMEA。"""
        super().__init__(
            P,
            population_size=population_size,
            elite_fraction=elite_fraction,
            statistical_weight=statistical_weight,
            graph_weight=graph_weight,
            max_linkage_size=max_linkage_size,
            max_linkage_sets=max_linkage_sets,
            seed=seed,
            device="cuda",
            cuda_device=cuda_device,
            require_cuda=require_cuda,
        )
        self.name = "GPU_GOMEA_WSN"
        self.acceleration_scope = "linkage_nmi"


CudaGOMEA = GPU_GOMEA
