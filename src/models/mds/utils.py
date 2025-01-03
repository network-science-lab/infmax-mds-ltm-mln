import warnings
from multiprocessing.managers import SharedMemoryManager
from  multiprocessing.shared_memory import ShareableList
from typing import Any

import network_diffusion as nd


class ShareableListManager:
    """A wrapper to SharableList used as a dump for temporary MDS solution."""

    placeholder_token = None

    def __init__(self, smm: SharedMemoryManager, length: int) -> None:
        self._sl = smm.ShareableList([self.placeholder_token] * length)

    @property
    def sl(self) -> ShareableList:
        return self._sl

    @sl.setter
    def sl(self, data: list[Any]) -> None:
        for idx in range(len(self._sl)):
            if idx < len(data):
                self._sl[idx] = data[idx]
            else:
                self._sl[idx] = self.placeholder_token

    def get_as_pruned_set(self) -> set[Any]:
        return set(item for item in self._sl if item != self.placeholder_token)


def is_dominating_set(candidate_ds: set[nd.MLNetworkActor], network: nd.MultilayerNetwork) -> bool:
    """Check whether provided candidate dominating set is in fact a dominating set."""
    # initialise a dictionary with nodes to dominate - start from putting there all nodes
    nodes_to_dominate = {
        l_name: {node for node in l_graph.nodes()} for l_name, l_graph in network.layers.items()
    }
    # iterate through all actors in the candidate dominating set
    for candidate_actor in candidate_ds:
        for l_name in candidate_actor.layers:
            for ca_neighbour in network.layers[l_name].neighbors(candidate_actor.actor_id):
                # if the neighbour of `candidate_actor` on layer `l_name` is still in the set of
                # nodes to be dominated remove it from there
                if ca_neighbour in nodes_to_dominate[l_name]:
                    nodes_to_dominate[l_name].remove(ca_neighbour)
    # check if in the set of undominated nodes are other nodes than these repres. dominating actors
    ca_ids = {ca.actor_id for ca in candidate_ds}
    for l_name in network.layers.keys():
        non_dominated_nodes = nodes_to_dominate[l_name].difference(ca_ids)
        if len(non_dominated_nodes) != 0:
            warnings.warn(
                f"Given set is not dominating - in {l_name} {non_dominated_nodes} are non-dominated"
            )
            return False
    return True
