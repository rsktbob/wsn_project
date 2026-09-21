import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.BaseEDA import BaseEDA
from Algorithm.ga.BaseGA import BaseGA
from Algorithm.gomea.BaseGIGOMEA import BaseGIGOMEA
from Algorithm.misc.BaseGWO import BaseGWO
from Algorithm.nsga.BaseNSGAII import BaseNSGAII
from Algorithm.misc.BasePSO import BasePSO
from Algorithm.se.BaseSE import BaseSE
from Algorithm.core.Algorithm import Algorithm, AlgorithmResult
from Algorithm.core import BatchEvaluator
from Algorithm.misc.ALNS import ALNS
from Algorithm.misc.CS import CS
from Algorithm.misc.EDA import EDA
from Algorithm.ga.GA import GA
from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from Algorithm.misc.GWO import GWO
from Algorithm.nsga.NSGAII import NSGAII
from Algorithm.misc.NSOA import NSOA
from Algorithm.misc.PSO import PSO
from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.SA_SETSv2 import SA_SETSv2
from Algorithm.se.SETSv1 import SETSv1
from Algorithm.se.SETSv2 import SETSv2
from Algorithm.Routing.CBGWO import CBGWO
from Algorithm.Routing.KPSOO import KPSOO
from Algorithm.Routing.RGA import RGA
from Algorithm.Routing.RQLearning import RQLearning
from Algorithm.Scheduling.NBEDA import NBEDA
from Algorithm.Scheduling.SES import SES
from Algorithm.Scheduling.SGA import SGA
from Algorithm.Scheduling.SNSGAII import SNSGAII
from Algorithm.Scheduling.SRIME import SRIME
from State.Encoding import Encoding
from State.SensorEncoding import SensorEncoding
from State.State import State
from State.TargetEncoding import TargetEncoding


def assert_not_implemented(call):
    try:
        call()
    except NotImplementedError:
        return
    raise AssertionError("Base hook must raise NotImplementedError")


class _State:
    def Copy(self):
        return self


class _Worker:
    def join(self):
        return None


class _Connection:
    def close(self):
        return None


class _MinimalBaseSE(BaseSE):
    """In-memory flow double used to verify the merged BaseSE engine contract."""

    def __init__(self):
        super().__init__(None, n=1, h=1, w=1)
        self.state = _State()
        self.vision_calls = 0
        self.market_calls = 0

    def _create_workers(self, P, iteration):
        return [_Worker()], [_Connection()]

    def _init_market(self, P):
        return [self.state], [1.0], [1.0]

    def _evaluate_states(self, P, states):
        return [1.0]

    def _invest(self, P, states, fitness, ta, tb, conn):
        self.vision_calls += 1
        return states, fitness

    def _update_beliefs(self, P, ta, tb):
        self.market_calls += 1
        return ta, tb

    def _evaluate_state(self, P, state):
        return [1.0]


def main():
    # Concrete, runnable algorithms are grouped by problem scope.
    combine_algorithms = (
        ALNS,
        CS,
        EDA,
        GA,
        GI_GOMEA,
        GI_GOMEA_Target,
        GWO,
        NSGAII,
        NSOA,
        PSO,
        SA_SETS,
        SA_SETSv2,
        SETSv1,
        SETSv2,
    )
    scheduling_algorithms = (NBEDA, SES, SGA, SNSGAII, SRIME)
    routing_algorithms = (CBGWO, KPSOO, RGA, RQLearning)

    organized_packages = {
        "ga",
        "gomea",
        "map_elites",
        "misc",
        "nsga",
        "se",
    }
    assert all(
        cls.__module__.split(".")[1] in organized_packages
        for cls in combine_algorithms
    )
    assert all(
        cls.__module__.startswith("Algorithm.Scheduling.")
        for cls in scheduling_algorithms
    )
    assert all(cls.__module__.startswith("Algorithm.Routing.") for cls in routing_algorithms)

    # Family algorithms reuse a small flow-owning base.
    assert issubclass(GA, BaseGA)
    assert issubclass(SGA, BaseGA)
    assert issubclass(RGA, BaseGA)
    assert issubclass(PSO, BasePSO)
    assert issubclass(KPSOO, BasePSO)
    assert issubclass(EDA, BaseEDA)
    assert issubclass(NBEDA, BaseEDA)
    assert issubclass(GWO, BaseGWO)
    assert issubclass(CBGWO, BaseGWO)
    assert issubclass(NSGAII, BaseNSGAII)
    assert issubclass(SNSGAII, BaseNSGAII)
    assert SA_SETS.__bases__ == (BaseSE,)
    assert SES.__bases__ == (BaseSE,)
    assert SETSv1.__bases__ == (SA_SETS,)
    assert SETSv2.__bases__ == (BaseSE,)
    assert issubclass(SA_SETSv2, SETSv2)
    assert CS.__bases__ == (Algorithm,)
    assert SRIME.__bases__ == (Algorithm,)

    # The generic bases fail early when a required problem-specific hook is absent.
    ga_base = BaseGA(n=2)
    assert_not_implemented(lambda: ga_base._create_state(None))
    assert_not_implemented(
        lambda: ga_base._crossover_states(None, None, 0, 1)
    )
    assert_not_implemented(lambda: ga_base._mutate_state(None, None))

    se_base = BaseSE(None, n=1, h=2, w=1)
    assert_not_implemented(lambda: se_base.create_candidate(None))
    assert_not_implemented(lambda: se_base.align_region(None, None, 0))
    assert_not_implemented(
        lambda: se_base.invest(None, None, None)
    )
    assert not hasattr(BaseSE, "_crossover_state")
    assert not hasattr(BaseSE, "_mutate_state")
    assert hasattr(BaseSE, "create_investments")
    assert hasattr(BaseSE, "select_regions")
    assert hasattr(BaseSE, "update_search_memory")
    assert se_base.iteration_callback is None
    assert se_base.iteration_listeners == []
    assert se_base.Fname == [
        "total_remaining_energy",
        "min_remaining_energy",
        "coverage",
    ]

    # Every optimizer can opt into the same on-iteration console reporter.
    report = StringIO()
    se_base.enable_iteration_reporting(every=2)
    with redirect_stdout(report):
        se_base.on_iteration_finish(
            iteration=1,
            run=0,
            best_value=[0.1, 0.2, 0.7],
        )
        se_base.on_iteration_finish(
            iteration=2,
            run=0,
            best_value=[0.1, 0.2, 0.7],
        )
    output = report.getvalue()
    assert "iteration=1" not in output
    assert "iteration=2" in output
    assert "total_remaining_energy=0.100000" in output
    assert "coverage=0.700000" in output
    assert "total=1.000000" in output
    se_base.disable_iteration_reporting()

    # GI-GOMEA target mode uses the retained target-coded state.
    assert GI_GOMEA.__module__ == "Algorithm.gomea.GI_GOMEA"
    assert GI_GOMEA_Target.__module__ == "Algorithm.gomea.GI_GOMEA_Target"
    assert GI_GOMEA.__bases__ == (BaseGIGOMEA,)
    assert GI_GOMEA_Target.__bases__ == (BaseGIGOMEA,)
    assert SensorEncoding.__name__ == "SensorEncoding"
    assert TargetEncoding.__name__ == "TargetEncoding"
    assert issubclass(SensorEncoding, Encoding)
    assert issubclass(TargetEncoding, Encoding)
    assert State.__name__ == "State"

    # 通用核心保持薄；族群平行評估由組合元件提供。
    assert AlgorithmResult.__name__ == "AlgorithmResult"
    assert BatchEvaluator.__module__ == "Algorithm.core.evaluation"
    assert not hasattr(Algorithm, "crossover_population")
    assert not hasattr(Algorithm, "_invest")
    for base in (
        BaseSE,
        BaseGA,
        BasePSO,
        BaseEDA,
        BaseGWO,
        BaseNSGAII,
        BaseGIGOMEA,
    ):
        assert not hasattr(base, "Run")
        assert not hasattr(base, "RunByEvaTime")
        assert not any(
            name[:1].isupper()
            for name, value in base.__dict__.items()
            if callable(value)
        )
    assert {
        BaseSE.__name__,
        BaseGA.__name__,
        BasePSO.__name__,
        BaseEDA.__name__,
        BaseGWO.__name__,
        BaseNSGAII.__name__,
        BaseGIGOMEA.__name__,
    } == {
        "BaseSE",
        "BaseGA",
        "BasePSO",
        "BaseEDA",
        "BaseGWO",
        "BaseNSGAII",
        "BaseGIGOMEA",
    }

    print("smoke_algorithm_architecture_ok")


if __name__ == "__main__":
    main()
