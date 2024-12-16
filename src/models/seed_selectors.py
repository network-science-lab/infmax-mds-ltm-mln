"""Functions with sed selectio methods not-yet implemented in network_diffusion."""

from typing import Any

import pandas as pd
import networkx as nx
import network_diffusion as nd

from bidict import bidict
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE
# from network_diffusion.mln.driver_actors import compute_driver_actors

from src.models.mds import is_dominating_set, compute_driver_actors

class DCBSelector(nd.seeding.BaseSeedSelector):
    """
    Degree-Closeness-Betweenness Seed Selector (DCB).
    
    We rank the actors by averaging the sum of each actor's degree, closeness and betweenness
    centrality values.
    """

    def __str__(self) -> str:
        return f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\nDCB\n{BOLD_UNDERLINE}\n"

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        raise NotImplementedError
    
    @staticmethod
    def get_mean_DCB(net: nd.mln.MultilayerNetwork) -> dict[nd.MLNetworkActor, float]:
        actor_map = bidict({actor: actor.actor_id for actor in net.get_actors()})
        degree = {k.actor_id: v for k, v in nd.mln.functions.degree(net).items()}
        closeness = {k.actor_id: v for k, v in nd.mln.functions.closeness(net).items()}
        betweenness = {k.actor_id: v for k, v in nd.mln.functions.betweenness(net).items()}
        aggr_vals = pd.DataFrame({"C": closeness, "B": betweenness, "D": degree})
        mean_vals = aggr_vals.mean(axis=1).to_dict()
        return {actor_map.inverse[actor_id]: mean_val for actor_id, mean_val in mean_vals.items()}

    def actorwise(self, net: nd.mln.MultilayerNetwork) -> list[nd.mln.MLNetworkActor]:
        ranking_dict = self.get_mean_DCB(net)
        return [actor for actor, _ in sorted(ranking_dict.items(), reverse=True, key=lambda x: x[1])]


class DriverActorLimitedSelector(nd.seeding.BaseSeedSelector):
    """Driver Actor based seed selector."""

    def __init__(self, method: nd.seeding.BaseSeedSelector | None, return_only_mds: bool = True) -> None:
        """
        Initialise object.

        :param method: a method to sort driver actors; if not provided it will be set to
            `nd.seeding.RandomSeedSelector()`.
        :param return_only_mds: a flag wether return only minimal dominating which is not padded to
            the lenght equals a total number of actors.
        """
        super().__init__()
        if isinstance(method, nd.seeding.DriverActorSelector):
            raise AttributeError(f"Argument 'method' cannot be {self.__class__.__name__}!")
        if not method:
            method = nd.seeding.RandomSeedSelector()
        self.selector = method
        self.return_only_mds = return_only_mds

    def __str__(self) -> str:
        slc = self.selector
        s_str = "driver actor" if slc is None else f"driver actor + {slc.__class__.__name__}"
        return f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n{s_str}\n{BOLD_UNDERLINE}\n"

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        raise NotImplementedError("Nodewise ranking list cannot be computed for this class!")

    def actorwise(self, net: nd.mln.MultilayerNetwork) -> list[nd.mln.MLNetworkActor]:
        """Return a list of driver actors for a multilayer network."""
        driver_actors = compute_driver_actors(net)
        if not is_dominating_set(candidate_ds=driver_actors, network=net):
            raise ValueError(
                f"A seed set: {set(a.actor_id for a in driver_actors)} does not dominate "
                f"a following network:\n {net}!"
            )

        all_actors_sorted = self.selector.actorwise(net)
        return self._reorder_seeds(driver_actors, all_actors_sorted)

    def _reorder_seeds(
        self, driver_actors: list[nd.mln.MLNetworkActor], all_actors: list[nd.mln.MLNetworkActor],
    ) -> list[nd.mln.MLNetworkActor]:
        """Return a list of actor ids, where driver actors in the first."""
        driver_actors_sorted, inferior_actors_sorted = [], []
        for actor in all_actors:
            if actor in driver_actors:
                driver_actors_sorted.append(actor)
            else:
                inferior_actors_sorted.append(actor)
        assert len(inferior_actors_sorted) + len(driver_actors_sorted) == len(all_actors)
        if self.return_only_mds:
            return driver_actors_sorted
        return [*driver_actors_sorted, *inferior_actors_sorted]


if __name__ == "__main__":
    net = nd.mln.functions.get_toy_network_piotr()
    print(net)
    # print(DCBSelector.get_mean_DCB(net))
    # print(DCBSelector()(net, actorwise=True))
    mds_selector = DriverActorLimitedSelector(method=nd.seeding.DegreeCentralitySelector())
    print(mds_selector)
    print(mds_selector(net, actorwise=True))
