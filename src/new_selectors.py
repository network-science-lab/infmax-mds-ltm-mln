"""Functions with sed selectio methods not-yet implemented in network_diffusion."""

from typing import Any

import pandas as pd
import networkx as nx
import network_diffusion as nd

from bidict import bidict
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


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


if __name__ == "__main__":
    net = nd.mln.functions.get_toy_network_piotr()
    print(net)
    print(DCBSelector.get_mean_DCB(net))
    print(DCBSelector()(net, actorwise=True))
