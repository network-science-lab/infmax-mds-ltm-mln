"""Script with functions for driver actor selections with local improvement."""

import random
from typing import Any, List, Set

import network_diffusion as nd


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


from src.models.mds.greedy_search import minimum_dominating_set_with_initial


def get_mds_locimpr(net: nd.MultilayerNetwork) -> List[nd.MLNetworkActor]:
    """Return driver actors for a given network using MDS and local improvement."""
    # Step 1: Compute initial Minimum Dominating Set
    initial_dominating_set: Set[Any] = set()
    for layer in net.layers:
        initial_dominating_set = minimum_dominating_set_with_initial(
            net, layer, initial_dominating_set
        )

    # Step 2: Apply Local Improvement to enhance the Dominating Set
    improved_dominating_set = local_improvement(net, initial_dominating_set)

    return [net.get_actor(actor_id) for actor_id in improved_dominating_set]


def local_improvement(net: nd.MultilayerNetwork, initial_set: set) -> set:
    """
    Perform local improvement on the initial dominating set using the First Improvement strategy,
    including the checking procedure after each feasible exchange move.
    """
    dominating_set = set(initial_set)

    # Precompute domination for each node
    domination = compute_domination(net, dominating_set)

    improvement = True
    while improvement:
        improvement = False
        # Shuffle the dominating set to diversify search of neighbors
        current_solution = list(dominating_set)
        random.shuffle(current_solution)

        for u in current_solution:
            # Identify candidate replacements v not in D, but only those leading to a feasible solution
            candidates_v = find_replacement_candidates(net, u, dominating_set, domination)
            random.shuffle(candidates_v)

            for v in candidates_v:
                # Store old solution for rollback if no improvement after checking
                old_dominating_set = set(dominating_set)

                # Attempt the exchange move
                new_dominating_set = (dominating_set - {u}) | {v}
                if is_feasible(net, new_dominating_set):
                    # After a feasible exchange, perform the checking procedure to remove redundancies
                    reduced_set = remove_redundant_vertices(net, new_dominating_set)

                    # Check if we actually improved (reduced the size of the solution)
                    if len(reduced_set) < len(old_dominating_set):
                        # We have found an improvement, update domination and break
                        dominating_set = reduced_set
                        domination = compute_domination(net, dominating_set)
                        improvement = True
                        break
                    else:
                        # No improvement after redundancy removal, revert to old solution
                        dominating_set = old_dominating_set
                        # domination stays the same, no improvement here
                # If not feasible, just continue trying other candidates

            if improvement:
                # Restart the outer loop after finding the first improvement
                break

    return dominating_set


def compute_domination(net: nd.MultilayerNetwork, dominating_set: Set[Any]) -> dict:
    """
    Compute the domination map for the current dominating set per layer.

    Returns a dictionary where keys are layer names and values are dictionaries
    mapping node IDs to sets of dominators in that layer.
    """
    domination_map = {
        layer: {actor.actor_id: set() for actor in net.get_actors()}
        for layer in net.layers
    }

    for layer, net_layer in net.layers.items():
        for actor_id in dominating_set:
            if actor_id in net_layer.nodes:
                domination_map[layer][actor_id].add(actor_id)  # A node dominates itself
                for neighbor in net_layer[actor_id]:
                    domination_map[layer][neighbor].add(actor_id)
    return domination_map


def find_replacement_candidates(net: nd.MultilayerNetwork, u: Any, dominating_set: Set[Any], domination: dict) -> List[
    Any]:
    """
    Find candidate nodes v that can replace u in the dominating set,
    ensuring that all layers remain dominated.
    """
    exclusively_dominated = {}

    for layer, net_layer in net.layers.items():
        if u in net_layer:
            exclusively_dominated[layer] = {
                w for w in set(net_layer[u]) | {u}
                if domination[layer][w] == {u}
            }
        else:
            exclusively_dominated[layer] = set()  # No nodes exclusively dominated by u in this layer

    # Find valid replacement candidates
    actor_ids = [x.actor_id for x in net.get_actors()]
    candidates = []
    for v in actor_ids:
        if v in dominating_set:
            continue

        # Ensure v exists in all layers where exclusively dominated nodes are expected
        if all(
                v in net.layers[layer] and nodes.issubset(set(net.layers[layer][v]) | {v})
                for layer, nodes in exclusively_dominated.items()
        ):
            candidates.append(v)

    return candidates


def is_feasible(net: nd.MultilayerNetwork, dominating_set: Set[Any]) -> bool:
    """
    Check if the dominating set is feasible across all layers.
    """
    for layer, net_layer in net.layers.items():
        dominated = set()
        for actor_id in dominating_set:
            if actor_id in net_layer.nodes:
                dominated.add(actor_id)
                dominated.update(net_layer[actor_id])
        if dominated != set(net_layer.nodes()):
            return False
    return True


def remove_redundant_vertices(net, dominating_set):
    """
    Try to remove redundant vertices from the dominating_set without losing feasibility.
    A vertex is redundant if removing it still leaves all nodes dominated.
    Returns a new dominating set with as many redundant vertices removed as possible.
    """
    # We'll attempt to remove vertices one by one.
    # A simple (although not necessarily minimum) approach is to try removing each vertex
    # and see if the set remains feasible. If yes, permanently remove it.
    improved = True
    improved_set = set(dominating_set)
    while improved:
        improved = False
        for d in list(improved_set):
            candidate_set = improved_set - {d}
            if is_feasible(net, candidate_set):
                improved_set = candidate_set
                improved = True
                # Break to re-check from scratch after every removal, ensuring first improvement strategy
                break
    return improved_set



if __name__ == "__main__":
    from utils import is_dominating_set
    l2c_1  = nd.tpn.get_l2_course_net(node_features=False, edge_features=False, directed=False)[0]
    # mds = nd.mln.driver_actors.compute_driver_actors(l2c_1)
    mds = get_mds_locimpr(l2c_1)
    # mds = set(l2c_1.get_actors())
    # mds.pop()
    if is_dominating_set(candidate_ds=mds, network=l2c_1):
        print(f"Given set: {set(ac.actor_id for ac in mds)} dominates the network!")
    else:
        print(f"Given set: {set(ac.actor_id for ac in mds)} does not dominate the network!")
