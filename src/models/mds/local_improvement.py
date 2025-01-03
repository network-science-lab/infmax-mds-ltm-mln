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
    improved_dominating_set = LocalImprovement(net)(initial_dominating_set)

    return [net.get_actor(actor_id) for actor_id in improved_dominating_set]


class LocalImprovement:
    """A class to prune initial Dominating Set."""

    def __init__(self, net: nd.MultilayerNetwork):
        self.net = net
        self.actors = net.get_actors()
        self.actor_ids = [x.actor_id for x in self.actors]
        self._cache = {}
        self._last_dominating_set = None

    def __call__(self,initial_set: set[Any]) -> set[Any]:
        return self.local_improvement(initial_set)

    def local_improvement(self, initial_set: set[Any]) -> set[Any]:
        """
        Perform local improvement on the initial dominating set using the First Improvement strategy,
        including the checking procedure after each feasible exchange move.
        """
        dominating_set = set(initial_set)

        # Precompute domination for each node
        domination = self._compute_domination(dominating_set)

        improvement = True
        while improvement:
            improvement = False
            # Shuffle the dominating set to diversify search of neighbors
            current_solution = list(dominating_set)
            random.shuffle(current_solution)

            for u in current_solution:
                # Identify candidate replacements v not in D, but only those leading to a feasible solution
                candidates_v = self._find_replacement_candidates(u, dominating_set, domination)
                random.shuffle(candidates_v)

                for v in candidates_v:
                    # Store old solution for rollback if no improvement after checking
                    old_dominating_set = set(dominating_set)

                    # Attempt the exchange move
                    new_dominating_set = (dominating_set - {u}) | {v}
                    if self._is_feasible(new_dominating_set):
                        # After a feasible exchange, perform the checking procedure to remove redundancies
                        reduced_set = self._remove_redundant_vertices(new_dominating_set)

                        # Check if we actually improved (reduced the size of the solution)
                        if len(reduced_set) < len(old_dominating_set):
                            # We have found an improvement, update domination and break
                            dominating_set = reduced_set
                            domination = self._compute_domination(dominating_set)
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


    def _compute_domination(self, dominating_set: set[Any]) -> dict[str, dict[Any, set[Any]]]:
        """
        Compute the domination map for the current dominating set per layer.

        Return a dictionary where keys are layer names and values are dictionaries mapping node IDs
        to sets of dominators in that layer.
        """
        if dominating_set == self._last_dominating_set:
            return self._cache

        domination_map = {
            layer: {actor.actor_id: set() for actor in self.actors} for layer in self.net.layers
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

    def _get_excusevely_dominated_by_u(
        self, u: Any, domination: dict[str, dict[Any, set[Any]]]
    ) -> dict[str, set[Any]]:
        """Get nodes that are exclusevely dominated by node u in the network."""
        ed = {}
        for layer, net_layer in self.net.layers.items():
            if u in net_layer:
                ed[layer] = {w for w in set(net_layer[u]) | {u} if domination[layer][w] == {u}}
            else:
                ed[layer] = set()  # No nodes exclusively dominated by u in this layer
        return ed

    def _find_replacement_candidates(
        self,
        u: Any,
        dominating_set: set[Any],
        domination: dict[str, dict[Any, set[Any]]],
    ) -> list[Any]:
        """
        Find candidate nodes v that can replace u in the dominating set, ensuring that all layers
        remain dominated.
        """
        exclusively_dominated = self._get_excusevely_dominated_by_u(u, domination)

        # Find valid replacement candidates
        candidates = []
        for v in self.actor_ids:
            if v in dominating_set:
                continue

            # Ensure v exists in all layers where exclusively dominated nodes are expected
            if all(
                    v in self.net.layers[layer]
                    and nodes.issubset(set(self.net.layers[layer][v]) | {v})
                    for layer, nodes in exclusively_dominated.items()
            ):
                candidates.append(v)

        return candidates

    def _is_feasible(self, dominating_set: set[Any]) -> bool:
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

    def _remove_redundant_vertices(self, dominating_set: set[Any]) -> set[any]:
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
                if self._is_feasible(candidate_set):
                    improved_set = candidate_set
                    under_improvement = True
                    # Break to re-check from scratch after every removal, ensuring first improvement strategy
                    break
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
