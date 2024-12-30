"""Functions to visualise MDS."""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import network_diffusion as nd
import pandas as pd

from src.visualisation import Results


ACTORS_COLOUR = "magenta"
MDS_ACTORS_COLOUR = "#210070"
ACTORS_SHAPE = "o"
CENTRAL_ACTORS_SHAPE = "*"
    

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
