"""A pipeline to generate and plot random networks with MDS."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import network_diffusion as nd
import uunet.multinet as ml

from src.network_generator import MultilayerERGenerator, MultilayerPAGenerator, draw_mds
from src.models.mds import greedy_search, local_improvement


def generate(model: Literal["PA", "ER"], nb_actors: int, nb_layers: int) -> None:
    """Generate a random multilayer Erdos-Renyi or Preferential-Attachement network."""
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
            nb_steps=nb_actors * 10,
            nb_hubs=nb_hubs,
        )()
    else:
        raise ValueError("Incorret model name!")
    print(net_uu)

    net_nx = ml.to_nx_dict(net_uu)
    print(net_nx)

    net_nd = nd.MultilayerNetwork(layers=net_nx)
    net_nd = nd.mln.functions.remove_selfloop_edges(net_nd)
    print(net_nd)
    return net_nd


def plot(net_nd: nd.MultilayerNetwork, mds: set[nd.MLNetworkActor], net_name: str, out_dir: Path):
    fig, axs = plt.subplots(nrows=1, ncols=len(net_nd.layers))
    for l_idx, (l_name, l_graph) in enumerate(net_nd.layers.items()):
        axs[l_idx].hist(nx.degree_histogram(l_graph), bins=10)
        axs[l_idx].set_title(l_name)
    fig.suptitle("Degree distribution")
    plt.savefig(out_dir / f"{net_name}_hist.png", dpi=300)
    draw_mds(net_nd, mds, out_dir / f"{net_name}_plot.png")


if __name__ == "__main__":

    out_dir = Path("./doodles")
    out_dir.mkdir(exist_ok=True, parents=True)

    net = generate("ER", 40, 3)
    mds = local_improvement.get_mds_locimpr(net)
    plot(net, mds, "ER", out_dir)

    net = generate("PA", 40, 3)
    mds = local_improvement.get_mds_locimpr(net)
    plot(net, mds, "PA", out_dir)
