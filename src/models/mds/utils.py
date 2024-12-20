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
