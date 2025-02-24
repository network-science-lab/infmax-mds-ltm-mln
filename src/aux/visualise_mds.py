"""Functions to visualise MDS."""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import network_diffusion as nd
import pandas as pd

from src.aux.slicer_plotter import ResultsSlicer
from src.aux import (
    MDS_ACTORS_COLOUR,
    NML_ACTORS_COLOUR,
    OTHER_ACTORS_COLOUR,
    ALL_ACTORS_SHAPE,
)

FONT_SIZE = 11
    

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
    

class MDSVisualiser:
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
        mds: set[Any],
        ax: matplotlib.axes.Axes,
    ):
        top_k = nodes_degrees.sort_values(by="degree", ascending=False)[:len(mds)].index
        nx.draw_networkx_nodes(
            graph,
            nodelist=[node_id],
            ax=ax,
            pos={node_id: nodes_pos[node_id]},
            node_size=200,
            edgecolors=MDS_ACTORS_COLOUR if node_id in mds else OTHER_ACTORS_COLOUR,
            node_color=NML_ACTORS_COLOUR if node_id in top_k else OTHER_ACTORS_COLOUR,
            node_shape=ALL_ACTORS_SHAPE,
            alpha=1,
            linewidths=3
        )

    @staticmethod
    def _get_centrality_threshold(degrees_df: pd.DataFrame, top_k: int, centrality_name: str) -> int:
        return degrees_df.iloc[top_k - 1][centrality_name]

    # def plot_structure(self) -> None:
    #     degrees_dict = {
    #         actor.actor_id: degree
    #         for actor, degree in nd.mln.functions.degree(self.net).items()
    #     }
    #     degrees_df = pd.DataFrame({"degree": degrees_dict}).sort_values("degree", ascending=False)
    #     degree_thresh = self._get_centrality_threshold(degrees_df, len(self.mds), "degree")
    #     mds_ids = {actor.actor_id for actor in self.mds}
    #     pos = nx.drawing.kamada_kawai_layout(squeeze_by_neighbourhood(self.net, False))
    #     fig, axs = plt.subplots(nrows=1, ncols=len(self.net.layers))
    #     if len(self.net.layers) == 1:  # fix for single-layered networks
    #         axs = [axs]
    #     for idx, layer_name in enumerate(sorted(self.net.layers.keys())):
    #         layer_graph = self.net.layers[layer_name]
    #         axs[idx].set_title(layer_name)
    #         nx.draw_networkx_edges(
    #             layer_graph,
    #             ax=axs[idx],
    #             pos=pos,
    #             alpha=0.25,
    #             width=2,
    #             edge_color=OTHER_ACTORS_COLOUR,
    #         )
    #         for node in layer_graph.nodes:
    #             self._draw_node(
    #                 graph=layer_graph,
    #                 node_id=node,
    #                 nodes_pos=pos,
    #                 nodes_degrees=degrees_df,
    #                 mds=mds_ids,
    #                 degree_thresh=degree_thresh,
    #                 ax=axs[idx],
    #             )
    #         nx.drawing.draw_networkx_labels(layer_graph, ax=axs[idx], pos=pos, font_size=5, alpha=1)
    #     fig.set_size_inches(15, 5)
    #     fig.tight_layout()
    #     plt.savefig(self.out_path / f"{self.name}_structure.pdf", dpi=300)

    def _plot_centrality(
        self,
        centr_name: str,
        centrality_vals: dict[str, int],
        cantrality_hist: dict[int, int],
        ax: matplotlib.axes.Axes,
    ) -> None:
        plt.rc("legend", fontsize=FONT_SIZE)
        ymax = max(cantrality_hist.values()) * 1.2
        xmax = max(cantrality_hist.keys()) * 1.2
        centr_mds = [centrality_vals[str(actor.actor_id)] for actor in self.mds]
        centr_df = pd.DataFrame({"centr": centrality_vals}).sort_values("centr", ascending=False)
        centr_thresh = self._get_centrality_threshold(centr_df, len(self.mds), "centr")
        ax.fill_between(
            x=[centr_thresh, xmax],
            y1=[ymax] * 2,
            color=NML_ACTORS_COLOUR,
            alpha=0.1,
        )
        ax.vlines(
            x=centr_mds,
            ymin=0,
            ymax=ymax,
            # label="D",
            colors=MDS_ACTORS_COLOUR,
            alpha= 1 / len(self.mds),
        )
        ax.vlines(
            x=centr_thresh,
            ymin=0,
            ymax=ymax, 
            # label="top(|D|, Aϕ)",
            colors=NML_ACTORS_COLOUR,
            alpha=.75,
        )
        ax.scatter(
            cantrality_hist.keys(),
            cantrality_hist.values(),
            marker="o",
            color=OTHER_ACTORS_COLOUR,
            alpha=1,
            label="",
        )
        ax.set_xlim(left=0, right=xmax, auto=True)
        ax.set_ylim(bottom=0, top=ymax, auto=True)
        c_legend = {MDS_ACTORS_COLOUR: "D", NML_ACTORS_COLOUR: "top(|D|, Aϕ)"}
        c_patch = [mpatches.Patch(color=color, label=label) for color, label in c_legend.items()]
        ax.legend(handles=c_patch)
        ax.set_title(f"{centr_name}", fontsize=FONT_SIZE)
        ax.set_xlabel("", fontsize=FONT_SIZE)
        ax.set_ylabel("", fontsize=FONT_SIZE)

    # def plot_centralities(self) -> None:
    #     fig, axs = plt.subplots(nrows=1, ncols=2)
    #     for idx, centr_name in enumerate(["degree", "neighbourhood_size"]):
    #         centr_dict = ResultsSlicer.prepare_centrality(self.net, centr_name)
    #         self._plot_centrality(
    #             centr_name=centr_name,
    #             centrality_vals=centr_dict[0],
    #             cantrality_hist=centr_dict[1],
    #             ax=axs[idx],
    #         )
    #     fig.set_size_inches(15, 5)
    #     fig.tight_layout()
    #     fig.suptitle(f"{self.name}")
    #     plt.savefig(self.out_path / f"{self.name}_centralities.pdf", dpi=300)
    
    def manuscript_plot(self):
        fig, axs = plt.subplots(nrows=1, ncols=len(self.net.layers) + 1)

        # structure
        degrees_dict = {
            actor.actor_id: degree
            for actor, degree in nd.mln.functions.degree(self.net).items()
        }
        degrees_df = pd.DataFrame({"degree": degrees_dict}).sort_values("degree", ascending=False)
        mds_ids = {actor.actor_id for actor in self.mds}
        squeezed_net = squeeze_by_neighbourhood(self.net, False)
        pos = nx.drawing.kamada_kawai_layout(squeezed_net)
        for idx, layer_name in enumerate(sorted(self.net.layers.keys())):
            layer_graph = self.net.layers[layer_name]
            axs[idx].set_title(layer_name, fontsize=FONT_SIZE)
            nx.draw_networkx_edges(
                layer_graph,
                ax=axs[idx],
                pos=pos,
                alpha=0.25,
                width=2,
                edge_color=OTHER_ACTORS_COLOUR,
            )
            for actor in squeezed_net.nodes():
                self._draw_node(
                    graph=layer_graph,
                    node_id=actor,
                    nodes_pos=pos,
                    nodes_degrees=degrees_df,
                    mds=mds_ids,
                    ax=axs[idx],
                )
            nx.drawing.draw_networkx_labels(squeezed_net, ax=axs[idx], pos=pos, font_size=5, alpha=1)
            c_legend = {MDS_ACTORS_COLOUR: "a ∈ D", NML_ACTORS_COLOUR: "a ∈ top(|D|, Aϕ)"}
            c_patch = [mpatches.Patch(color=color, label=label) for color, label in c_legend.items()]
            axs[idx].legend(handles=c_patch, fontsize=FONT_SIZE)

        # degree
        centr_dict = ResultsSlicer.prepare_centrality(self.net, "degree")
        self._plot_centrality(
            centr_name="degree distribution",
            centrality_vals=centr_dict[0],
            cantrality_hist=centr_dict[1],
            ax=axs[-1],
        )

        # figure's details
        fig.set_size_inches(12, 4)
        fig.tight_layout()
        plt.savefig(self.out_path / f"{self.name}_manuscript.pdf", dpi=300)
        print(f"{self.name}: MDS size: {len(self.mds)}, number actors: {self.net.get_actors_num()}")
