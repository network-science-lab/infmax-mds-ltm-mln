"""Generators of multilayer networks with Preferential Attachment or Erdos-Renyi models."""

import abc
import warnings
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd
import uunet.multinet as ml
from uunet._multinet import PyMLNetwork, PyEvolutionModel

from src.visualisation import Results


ACTORS_COLOUR = "magenta"
MDS_ACTORS_COLOUR = "#210070"
ACTORS_SHAPE = "o"
CENTRAL_ACTORS_SHAPE = "*"


class MultilayerBaseGenerator(abc.ABC):
    """Abstract base class for multilayer network generators."""

    @abc.abstractmethod
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        """
        self.nb_layers=nb_layers
        self.nb_actors=nb_actors
        self.nb_steps=nb_steps

    @abc.abstractmethod
    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of concrete network generators
        """
        pass

    def get_dependency(self) -> np.ndarray:
        """
        Create a dependency matrix for the network generator.

        :return: matrix shaped: [nb_layers, nb_layers] with 0s in diagonal and 1/nb_layers otherwise
        """
        dep = np.full(fill_value=(1 / self.nb_layers), shape=(self.nb_layers, self.nb_layers))
        np.fill_diagonal(dep, 0)
        return dep

    def __call__(self) -> PyMLNetwork:
        """
        Generate the network according to given parameters.

        :return: generated network as a PyEvolutionModel object
        """
        models = self.get_models()
        pr_internal = np.random.uniform(0, 1, size=self.nb_layers)
        pr_external = np.ones_like(pr_internal) - pr_internal  # TODO: there is no noaction step now!
        dependency = self.get_dependency()
        return ml.grow(
            self.nb_actors,
            self.nb_steps,
            models,
            pr_internal.tolist(),
            pr_external.tolist(),
            dependency.tolist(),
        )


class MultilayerERGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Erdos-Renyi algorithm."""
    
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int, std_nodes: int) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param std_nodes: standard deviation of the number of nodes in each layer (expected value
            is a number of actors)
        """
        super().__init__(nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps)
        assert nb_actors - 2 * std_nodes > 0, "Number of actors shall be g.t. 2 * std_nodes"
        self.std_nodes=std_nodes

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer with num nodes drawn from a standard distribution.

        :return: list of Erdos-Renyi generators
        """
        layer_sizes = np.random.normal(
            loc=self.nb_actors - self.std_nodes, scale=self.std_nodes, size=self.nb_layers
        ).astype(np.int32).clip(min=1, max=self.nb_actors)
        return [ml.evolution_er(n=lv) for lv in layer_sizes]


class MultilayerPAGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Preferential Attachment algorithm."""
    
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int, nb_hubs: int) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param nb_seeds: number of seeds in each layer and a number of egdes from each new vertex
        """
        super().__init__(nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps)
        self.nb_hubs = nb_hubs

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of Preferential Attachment generators
        """
        m0s = [self.nb_hubs] * self.nb_layers
        ms = m0s.copy()
        return [ml.evolution_pa(m0=m0, m=ms) for m0, ms in zip(m0s, ms)]
    

# TODO: use nd based function once it gets updated to the current form!
def squeeze_by_neighbourhood(
    net: nd.MultilayerNetwork, preserve_mlnetworkactor_objects: bool = True
) -> nx.Graph:
    """Squeeze multilayer network to a single layer `nx.Graph`."""
    squeezed_net = nx.Graph()
    for actor in net.get_actors():
        for neighbour in nd.mln.functions.all_neighbors(net, actor):
            edge = (
                (actor, neighbour)
                if preserve_mlnetworkactor_objects
                else (actor.actor_id, neighbour.actor_id)
            )
            squeezed_net.add_edge(*edge)
        squeezed_net.add_node(actor if preserve_mlnetworkactor_objects else actor.actor_id)
    assert net.get_actors_num() == len(squeezed_net)
    return squeezed_net
    

class MDSPlotter:
    """Visualise MDS on the network's structure and on centralitiy distributions."""

    def __init__(
        self,
        net: nd.MultilayerNetwork,
        mds: set[nd.MLNetworkActor],
        name: str,
        out_path: str,
        ) -> None:
        self.net = net
        self.mds = mds
        self.name = name
        self.out_path = out_path

    @staticmethod
    def _draw_node(
        graph: nx.Graph,
        node_id: Any,
        nodes_pos: dict[Any, tuple[float, float]],
        nodes_degrees: pd.DataFrame,
        degree_thresh: int,
        mds: set[Any],
        ax: matplotlib.axes.Axes,
    ):
        nx.draw_networkx_nodes(
            graph,
            nodelist=[node_id],
            ax=ax,
            pos={node_id: nodes_pos[node_id]},
            node_size=150,
            node_color=MDS_ACTORS_COLOUR if node_id in mds else ACTORS_COLOUR,
            node_shape=(
                CENTRAL_ACTORS_SHAPE if nodes_degrees.loc[node_id]["degree"] >= degree_thresh
                else ACTORS_SHAPE
            ),
            alpha=0.5,
        )

    @staticmethod
    def _get_centrality_threshold(degrees_df: pd.DataFrame, top_k: int, centrality_name: str) -> int:
        return degrees_df.iloc[top_k - 1][centrality_name]

    def plot_structure(self) -> None:
        degrees_dict = {
            actor.actor_id: degree
            for actor, degree in nd.mln.functions.degree(self.net).items()
        }
        degrees_df = pd.DataFrame({"degree": degrees_dict}).sort_values("degree", ascending=False)
        degree_thresh = self._get_centrality_threshold(degrees_df, len(self.mds), "degree")
        mds_ids = {actor.actor_id for actor in self.mds}
        pos = nx.drawing.kamada_kawai_layout(squeeze_by_neighbourhood(self.net, False))
        fig, axs = plt.subplots(nrows=1, ncols=len(self.net.layers))
        for idx, (layer_name, layer_graph) in enumerate(self.net.layers.items()):
            axs[idx].set_title(layer_name)
            nx.draw_networkx_edges(
                layer_graph,
                ax=axs[idx],
                pos=pos,
                alpha=0.25,
                width=1,
                edge_color=ACTORS_COLOUR,
            )
            for node in layer_graph.nodes:
                self._draw_node(
                    graph=layer_graph,
                    node_id=node,
                    nodes_pos=pos,
                    nodes_degrees=degrees_df,
                    mds=mds_ids,
                    degree_thresh=degree_thresh,
                    ax=axs[idx],
                )
            nx.drawing.draw_networkx_labels(layer_graph, ax=axs[idx], pos=pos, font_size=5, alpha=1)
        fig.set_size_inches(15, 5)
        fig.tight_layout()
        fig.suptitle(f"{self.name}")
        plt.savefig(self.out_path / f"{self.name}_structure.png", dpi=300)

    def _plot_centrality(
        self,
        centr_name: str,
        centrality_vals: dict[str, int],
        cantrality_hist: dict[int, int],
        ax: matplotlib.axes.Axes,
    ) -> None:
        plt.rc("legend", fontsize=8)
        ymax = max(cantrality_hist.values()) * 1.2
        centr_mds = [centrality_vals[str(actor.actor_id)] for actor in self.mds]
        centr_df = pd.DataFrame({"centr": centrality_vals}).sort_values("centr", ascending=False)
        centr_thresh = self._get_centrality_threshold(centr_df, len(self.mds), "centr")
        ax.scatter(cantrality_hist.keys(), cantrality_hist.values(), marker="o", color=ACTORS_COLOUR)
        ax.vlines(x=centr_mds, ymin=0, ymax=ymax, label="MDS", colors=MDS_ACTORS_COLOUR, alpha=1)
        ax.vlines(x=centr_thresh, ymin=0, ymax=ymax, label="top-k threshold", colors="red", alpha=1)
        ax.set_xlim(left=0, auto=True)
        ax.set_ylim(bottom=0, top=ymax, auto=True)
        ax.yaxis.set_visible(False)
        ax.legend(loc="upper right")
        ax.set_title(f"{centr_name}")

    def plot_centralities(self) -> None:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        for idx, centr_name in enumerate(["degree", "neighbourhood_size"]):
            centr_dict = Results.prepare_centrality(self.net, centr_name)
            self._plot_centrality(
                centr_name=centr_name,
                centrality_vals=centr_dict[0],
                cantrality_hist=centr_dict[1],
                ax=axs[idx],
            )
        fig.set_size_inches(15, 5)
        fig.tight_layout()
        fig.suptitle(f"{self.name}")
        plt.savefig(self.out_path / f"{self.name}_centralities.png", dpi=300)
