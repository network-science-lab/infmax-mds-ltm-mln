"""Script with functions for driver actor selections with local improvement."""

import random
import time
from typing import Any

import network_diffusion as nd
import sys
from pathlib import Path

try:
    import src
except:
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    print(sys.path)

from src.models.mds.greedy_search import minimum_dominating_set_with_initial


def get_mds_locimpr(net: nd.MultilayerNetwork) -> list[nd.MLNetworkActor]:
    """Return driver actors for a given network using MDS and local improvement."""
    # Step 1: Compute initial Minimum Dominating Set
    initial_dominating_set: set[Any] = set()
    for layer in net.layers:
        initial_dominating_set = minimum_dominating_set_with_initial(
            net, layer, initial_dominating_set
        )

    # Step 2: Apply Local Improvement to enhance the Dominating Set
    improved_dominating_set = local_improvement(net, initial_dominating_set)

    return [net.get_actor(actor_id) for actor_id in improved_dominating_set]


def local_improvement(net: nd.MultilayerNetwork, initial_set: set[Any]) -> set[Any]:
    """
    Perform local improvement on the initial dominating set using the First Improvement strategy,
    including the checking procedure after each feasible exchange move.
    """
    dominating_set = set(initial_set)

    # Precompute domination for each node
    compute_domination = ComputeDomination(net=net)
    domination = compute_domination(dominating_set)

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
                        domination = compute_domination(dominating_set)
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


from concurrent.futures import ThreadPoolExecutor, as_completed


# def local_improvement2(net: nd.MultilayerNetwork, initial_set: set[Any]) -> set[Any]:
#     """
#     Perform local improvement on the initial dominating set using the First Improvement strategy,
#     including the checking procedure after each feasible exchange move.
#     """
#     dominating_set = set(initial_set)

#     # Precompute domination for each node
#     compute_domination = ComputeDomination(net=net)
#     domination = compute_domination(dominating_set)

#     improvement = True
#     while improvement:
#         improvement = False

#         # Shuffle the dominating set to diversify search of neighbors
#         current_solution = list(dominating_set)
#         random.shuffle(current_solution)

#         for u in current_solution:
#             # Step 1: Find candidates for replacing `u`
#             candidates_v = find_replacement_candidates(net, u, dominating_set, domination)
#             random.shuffle(candidates_v)

#             # Step 2: Define a function to evaluate a candidate
#             def evaluate_candidate(v):
#                 """
#                 Check if replacing `u` with `v` improves the dominating set.
#                 Returns the updated dominating set and a flag indicating improvement.
#                 """
#                 new_dominating_set = (dominating_set - {u}) | {v}
#                 if is_feasible(net, new_dominating_set):
#                     reduced_set = remove_redundant_vertices(net, new_dominating_set)
#                     if len(reduced_set) < len(dominating_set):
#                         return reduced_set, True
#                 return dominating_set, False

#             # Step 3: Submit tasks to the thread pool and track the futures
#             with ThreadPoolExecutor() as executor:
#                 futures = {executor.submit(evaluate_candidate, v): v for v in candidates_v}

#                 # Step 4: Check for the first improvement and cancel remaining tasks
#                 for future in as_completed(futures):
#                     new_set, improved = future.result()
#                     if improved:
#                         # Cancel all remaining tasks as we found an improvement
#                         for other_future in futures:
#                             other_future.cancel()

#                         # Update domination
#                         dominating_set = new_set
#                         domination = compute_domination(dominating_set)
#                         improvement = True

#                         break

#             if improvement:
#                 break  # Restart the outer loop after finding an improvement


#     return dominating_set










class ComputeDomination:
    """Compute the domination map for the current dominating set per layer."""

    def __init__(self, net: nd.MultilayerNetwork):
        self.net = net
        self.actors = net.get_actors()
        self._cache = {}
        self._last_dominating_set = None

    def __call__(self, dominating_set: set[Any]) -> dict:
        """
        Return a dictionary where keys are layer names and values are dictionaries mapping node IDs
        to sets of dominators in that layer.
        """
        if dominating_set == self._last_dominating_set:
            return self._cache

        domination_map = {
            layer: {actor.actor_id: set() for actor in self.actors}
            for layer in self.net.layers
        }
        for l_name, l_graph in self.net.layers.items():
            for actor_id in dominating_set:
                if actor_id in l_graph.nodes:
                    domination_map[l_name][actor_id].add(actor_id)  # a node dominates itself
                    for neighbour in l_graph[actor_id]:
                        domination_map[l_name][neighbour].add(actor_id)
        self._cache = domination_map
        self._last_dominating_set = set(dominating_set)
        return domination_map



def find_replacement_candidates(net: nd.MultilayerNetwork, u: Any, dominating_set: set[Any], domination: dict) -> list[
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


# from concurrent.futures import ThreadPoolExecutor
# from typing import List


# def find_replacement_candidates(net: nd.MultilayerNetwork, u: Any, dominating_set: set[Any], domination: dict) -> list[
#     Any]:
#     """
#     Find candidate nodes v that can replace u in the dominating set,
#     ensuring that all layers remain dominated.
#     """

#     # Step 1: Precompute nodes exclusively dominated by `u` in each layer
#     exclusively_dominated = {}
#     for layer, net_layer in net.layers.items():
#         if u in net_layer:
#             exclusively_dominated[layer] = {
#                 w for w in set(net_layer[u]) | {u}
#                 if domination[layer][w] == {u}
#             }
#         else:
#             exclusively_dominated[layer] = set()  # No nodes exclusively dominated by u in this layer


#     # Step 2: Define the candidate validation function
#     def is_valid_candidate(v: Any) -> bool:
#         """Check if `v` can replace `u` in the dominating set. """
#         # print(v)
#         if v in dominating_set:
#             return False
#         return all(
#             v in net.layers[layer] and nodes.issubset(set(net.layers[layer][v]) | {v})
#             for layer, nodes in exclusively_dominated.items()
#         )

#     actor_ids = [x.actor_id for x in net.get_actors()]

#     # Step 3: Parallelize the candidate evaluation
#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(is_valid_candidate, actor_ids))

#     # Step 4: Collect valid candidates
#     return [v for v, valid in zip(actor_ids, results) if valid]



def is_feasible(net: nd.MultilayerNetwork, dominating_set: set[Any]) -> bool:
    """Check if the dominating set is feasible across all layers."""
    for _, l_graph in net.layers.items():
        dominated = set()
        for actor_id in dominating_set:
            if actor_id in l_graph.nodes:
                dominated.add(actor_id)
                dominated.update(l_graph[actor_id])
        if dominated != set(l_graph.nodes()):
            return False
    return True


def remove_redundant_vertices2(net: nd.MultilayerNetwork, dominating_set: set[Any]) -> set[any]:
    """
    Try to remove redundant vertices from the dominating_set without losing feasibility.

    A vertex is redundant if removing it still leaves all nodes dominated.
    Returns a new dominating set with as many redundant vertices removed as possible.
    We'll attempt to remove vertices one by one. A simple (although not necessarily minimum)
    approach is to try removing each vertex and see if the set remains feasible. If yes,
    permanently remove it.
    """
    improved_set = set(dominating_set)
    under_improvement = True
    while under_improvement:
        under_improvement = False
        for d in improved_set:
            candidate_set = improved_set - {d}
            if is_feasible(net, candidate_set):
                improved_set = candidate_set
                under_improvement = True
                # Break to re-check from scratch after every removal, ensuring first improvement strategy
                break
    return improved_set


from concurrent.futures import ThreadPoolExecutor, as_completed

def remove_redundant_vertices(net: nd.MultilayerNetwork, dominating_set: set[Any]) -> set[Any]:
    """
    Try to remove redundant vertices from the dominating_set without losing feasibility.

    A vertex is redundant if removing it still leaves all nodes dominated.
    Returns a new dominating set with as many redundant vertices removed as possible.
    We'll attempt to remove vertices one by one. A simple (although not necessarily minimum)
    approach is to try removing each vertex and see if the set remains feasible. If yes,
    permanently remove it.
    """
    improved_set = set(dominating_set)
    under_improvement = True

    def can_remove(vertex: Any) -> tuple[Any, bool]:
        # print(f"Checking vertex: {vertex}")
        candidate_set = improved_set - {vertex}
        return vertex, is_feasible(net, candidate_set)

    while under_improvement:
        candidate_vertices = list(improved_set)
        removable_vertex = None
    
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(can_remove, vertex) for vertex in candidate_vertices]
            for future in as_completed(futures):
                vertex, removable = future.result()
                if removable:
                    for other_future in futures:
                        other_future.cancel()
                    removable_vertex = vertex
                    break

        if removable_vertex:
            improved_set.remove(removable_vertex)
            under_improvement = True
        else:
            under_improvement = False

    return improved_set



if __name__ == "__main__":
    # to run this example update PYTHONPATH
    from utils import is_dominating_set
    from src.loaders.net_loader import load_network
    from src.models.mds.greedy_search import get_mds_greedy

    net = load_network("sf2", as_tensor=False)
    # net = load_network("ckm_physicians", as_tensor=False)

    start_time = time.time()
    mds = get_mds_locimpr(net)
    # mds = get_mds_greedy(net)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # mds.pop()
    if is_dominating_set(candidate_ds=mds, network=net):
        print(f"A {len(mds)}-length set: {set(ac.actor_id for ac in mds)} dominates the network!")
    else:
        print(f"A {len(mds)}-length set: {set(ac.actor_id for ac in mds)} does not dominate the network!")
