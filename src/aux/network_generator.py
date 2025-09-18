"""
Generators of multilayer networks - Preferential Attachment or Erdos-Renyi.

"documentation": github.com/uuinfolab/r_multinet/blob/master/man/Generation.Rd
also, refer to this paper: `Formation of Multiple Networks` by Magnani and Rossa

This is almost the same script as in network-diffusion v.0.18.1
"""

import abc
from typing import Literal

import numpy as np
import uunet.multinet as ml
from uunet._multinet import PyEvolutionModel, PyMLNetwork

from network_diffusion.mln import functions
from network_diffusion.mln.mlnetwork import MultilayerNetwork


class MultilayerBaseGenerator(abc.ABC):
    """Abstract base class for multilayer network generators."""

    @abc.abstractmethod
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which
            builds the network
        """
        self.nb_layers = nb_layers
        self.nb_actors = nb_actors
        self.nb_steps = nb_steps

    @abc.abstractmethod
    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of concrete network generators
        """
        pass

    def get_pr_matrices(self) -> tuple[list[float], list[float]]:
        """
        Create probability matrices at random.

        `pr_internal` denotes probability of generation step within a layer
        `pr_external` denotes probability of copying edge from another layer
            according to `dependency` matrix
        `pr_no_action` is a resulting probability of no action step and its a 
            matrix of ones with above matrices subtracted from it.

        :return: created matrices in the order: internal, external
        """
        pr_internal = np.random.uniform(0, 1, size=self.nb_layers).tolist()
        pr_external = (np.ones_like(pr_internal) - pr_internal).tolist()
        return pr_internal, pr_external

    def get_dependency(self) -> np.ndarray:
        """
        Create a dependency matrix for the network generator.

        It generates a matrix that treats all layers equally and it's used if 
        external step occurs during network creation.

        :return: matrix shaped: [nb_layers, nb_layers] with 0s in diagonal
            and 1/(nb_layers-1) otherwise
        """
        dep = np.full(
            fill_value=(1 / (self.nb_layers - 1)),
            shape=(self.nb_layers, self.nb_layers),
        )
        np.fill_diagonal(dep, 0)
        return dep

    def __call__(
        self,
        pr_internal: list[float] | None = None,
        pr_external: list[float] | None = None,
        dependency: list[list[float]] | None = None,
    ) -> PyMLNetwork:
        """
        Generate the network according to given parameters.

        :return: generated network as a PyEvolutionModel object
        """
        models = self.get_models()
        if not pr_internal or not pr_internal:
            pr_internal, pr_external = self.get_pr_matrices()
        else:
            assert (
                np.array(pr_internal) + np.array(pr_external)
                <= np.ones(len(models))
            ).all()
        if not dependency:
            dependency = self.get_dependency().tolist()
        else:
            np.testing.assert_almost_equal(
                np.array(dependency).sum(axis=1),
                np.ones(len(dependency)),
            )
        return ml.grow(
            self.nb_actors,
            self.nb_steps,
            models,
            pr_internal,
            pr_external,
            dependency,
        )


class MultilayerERGenerator(MultilayerBaseGenerator):
    """Erdos-Renyi multilayer networks generator."""

    def __init__(
        self, nb_layers: int, nb_actors: int, nb_steps: int, std_nodes: int
    ) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which
            builds the network
        :param std_nodes: standard deviation of the number of nodes in each
            layer (expected value is a number of actors)
        """
        super().__init__(
            nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps
        )
        assert (
            nb_actors - 2 * std_nodes > 0
        ), "Number of actors shall be g.t. 2 * std_nodes"
        self.std_nodes = std_nodes

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get the the evolutionary  model.

        For each layer with num nodes drawn from a standard distribution.

        :return: list of Erdos-Renyi generators
        """
        layer_sizes = (
            np.random.normal(
                loc=self.nb_actors - self.std_nodes,
                scale=self.std_nodes,
                size=self.nb_layers,
            )
            .astype(np.int32)
            .clip(min=1, max=self.nb_actors)
        )
        return [ml.evolution_er(n=lv) for lv in layer_sizes]


class MultilayerPAGenerator(MultilayerBaseGenerator):
    """Preferential Attachment multilayer networks generator."""

    def __init__(
        self, nb_layers: int, nb_actors: int, nb_steps: int, nb_hubs: int
    ) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which
            builds the network
        :param nb_seeds: number of seeds in each layer and a number of egdes
            from each new vertex
        """
        super().__init__(
            nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps
        )
        self.nb_hubs = nb_hubs

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of Preferential Attachment generators
        """
        m0s = [self.nb_hubs] * self.nb_layers
        ms = m0s.copy()
        return [ml.evolution_pa(m0=m0, m=ms) for m0, ms in zip(m0s, ms)]


def convert_to_nd_and_prune(net_uu: PyMLNetwork) -> MultilayerNetwork:
    """Convert `uunet` representation to `networkx` and to `network_diffusion`."""
    net_nx = ml.to_nx_dict(net_uu)
    net_nd = MultilayerNetwork(layers=net_nx)

    # remove selfloops from the network and actors with no connections
    net_nd = functions.remove_selfloop_edges(net_nd)
    actors_to_remove = [
        ac.actor_id
        for ac, nghb in functions.neighbourhood_size(net_nd).items()
        if nghb == 0
    ]
    for l_graph in net_nd.layers.values():
        l_graph.remove_nodes_from(actors_to_remove)

    return net_nd


def save_as_mpx(net_nd: MultilayerNetwork, path: str) -> None:
    """Save the multilayer network as an mpx file."""
    dummy_net = ml.empty()
    for l_name in net_nd.layers:
        ml.add_nx_layer(dummy_net, net_nd[l_name], l_name)
    ml.write(dummy_net, path, "multilayer")


def generate(
    model: Literal["PA", "ER"], nb_actors: int, nb_layers: int
) -> MultilayerNetwork:
    """Generate a multilayer Erdos-Renyi or Preferential-Attachement net."""
    if model == "ER":
        std_nodes = int(0.1 * nb_actors)
        net_uu = MultilayerERGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors * 10,
            std_nodes=std_nodes,
        )()
    elif model == "PA":
        nb_hubs = int(max(2, 0.05 * nb_actors))
        net_uu = MultilayerPAGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors - nb_hubs,
            nb_hubs=nb_hubs,
        )()
    else:
        raise ValueError("Incorrect model name!")

    return convert_to_nd_and_prune(net_uu)


if __name__ == "__main__":
    nb_layers=3
    nb_actors=500
    nb_hubs=5
    pr_internal = [.7, .7, .7]
    pr_external = [.2, .2, .2]
    out_path = "./network.mpx"

    net_uu = MultilayerPAGenerator(
        nb_layers=nb_layers,
        nb_actors=nb_actors,
        nb_steps=nb_actors - nb_hubs,
        nb_hubs=nb_hubs,
    )(
        pr_internal=pr_internal,
        pr_external=pr_external,
    )
    net_nd = convert_to_nd_and_prune(net_uu)
    save_as_mpx(net_nd, out_path)

    print(net_nd)
    net_in = MultilayerNetwork.from_mpx(out_path)
    print(net_in)
