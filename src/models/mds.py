import warnings

import network_diffusion as nd


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


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Contents copied from network_diffusion 0.17.0 and modified with actor-shuffling step

import random
from copy import deepcopy
from typing import Any


def compute_driver_actors(net: nd.MultilayerNetwork) -> list[nd.MLNetworkActor]:
    """Return driver actors for a given network."""
    min_dominating_set: set[Any] = set()
    for layer in net.layers:
        min_dominating_set = minimum_dominating_set_with_initial(
            net, layer, min_dominating_set
        )

    return [net.get_actor(actor_id) for actor_id in min_dominating_set]


def minimum_dominating_set_with_initial(
    net: nd.MultilayerNetwork, layer: str, initial_set: set[Any]
) -> set[int]:
    """
    Return a dominating set that includes the initial set.

    net: MultilayerNetwork
    layer: layer name
    initial_set: set of nodes
    """
    actor_ids = [x.actor_id for x in net.get_actors()]
    random.shuffle(actor_ids)
    if not set(initial_set).issubset(set(actor_ids)):
        raise ValueError("Initial set must be a subset of net's actors")

    dominating_set = set(initial_set)

    net_layer = net.layers[layer]
    isolated = set(actor_ids) - set(net_layer.nodes())
    dominating_set = dominating_set | isolated
    dominated = deepcopy(dominating_set)

    layer_nodes = list(net_layer.nodes())
    random.shuffle(layer_nodes)

    for node_u in dominating_set:
        if node_u in net_layer.nodes:
            dominated.update(net_layer[node_u])

    while len(dominated) < len(net):
        node_dominate_nb = {x: len(set(net_layer[x]) - dominated) for x in layer_nodes}

        # If current dominated set cannot be enhanced anymore and there're still nondominated nodes
        if sum(node_dominate_nb.values()) == 0:
            to_dominate = set(net[layer].nodes).difference(dominated)
            return dominating_set.union(to_dominate)

        # Choose a node which dominates the maximum number of undominated nodes
        node_u = max(node_dominate_nb.keys(), key=lambda x: node_dominate_nb[x])
        dominating_set.add(node_u)
        dominated.add(node_u)
        dominated.update(net_layer[node_u])

    return dominating_set

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####


if __name__ == "__main__":
    l2c_1  = nd.tpn.get_l2_course_net(node_features=False, edge_features=False, directed=False)[0]
    # mds = nd.mln.driver_actors.compute_driver_actors(l2c_1)
    mds = compute_driver_actors(l2c_1)
    # mds = set(l2c_1.get_actors())
    # mds.pop()
    if is_dominating_set(candidate_ds=mds, network=l2c_1):
        print(f"Given set: {set(ac.actor_id for ac in mds)} dominates the network!")
    else:
        print(f"Given set: {set(ac.actor_id for ac in mds)} does not dominate the network!")
