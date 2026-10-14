from Algorithm.nsga.BaseNSGAII import BaseNSGAII
from Algorithm.Routing.RQLearning import RQLearning
from State.TargetEncoding import TargetEncoding


class SNSGAII(BaseNSGAII):
    """NSGA-II scheduler with optional Q-learning next-hop routing."""

    def __init__(
        self,
        n=50,
        generation=100,
        cu=0.9,
        mu=None,
        tournament_size=2,
        route_selector=None,
        use_rqlearning=True,
        rqlearning_kwargs=None,
    ):
        if route_selector is None and use_rqlearning:
            route_selector = RQLearning(**(rqlearning_kwargs or {}))

        self.route_selector = route_selector
        coding_kwargs = {}
        if route_selector is not None:
            coding_kwargs["route_selector"] = route_selector

        super().__init__(
            n=n,
            generation=generation,
            cu=cu,
            mu=mu,
            tournament_size=tournament_size,
            coding_cls=TargetEncoding,
            coding_kwargs=coding_kwargs,
        )
        self.name = "SNSGAII_" + str(n)

    def set_route_selector(self, route_selector):
        self.route_selector = route_selector
        if route_selector is None:
            self.coding_kwargs.pop("route_selector", None)
        else:
            self.coding_kwargs["route_selector"] = route_selector
        return self

    def create_route_selector(self, **kwargs):
        return self.set_route_selector(RQLearning(**kwargs))
