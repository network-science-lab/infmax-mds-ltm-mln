"""A pipeline to generate and plot random networks with MDS."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import network_diffusion as nd
import uunet.multinet as ml

from src.network_generator import MultilayerERGenerator, MultilayerPAGenerator, draw_mds


def generate(model: Literal["PA", "ER"], nb_actors: int, nb_layers: int, out_dir: Path) -> None:
    """Generate a random network and plot it with MDS."""
    if model == "ER":
        std_nodes = int(0.1 * nb_actors)
        net_uu = MultilayerERGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors * 5,
            std_nodes=std_nodes,
        )()
    elif model == "PA":
        nb_hubs = int(max(2, 0.05 * nb_actors))
        net_uu = MultilayerPAGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors * 5,
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

    fig, axs = plt.subplots(nrows=1, ncols=len(net_nx))
    for l_idx, (l_name, l_graph) in enumerate(net_nx.items()):
        axs[l_idx].hist(nx.degree_histogram(l_graph), bins=10)
        axs[l_idx].set_title(l_name)
    fig.suptitle("Degree distribution")
    plt.savefig(out_dir / f"{model}_hist.png", dpi=300)

    draw_mds(net_nd, {}, out_dir / f"{model}_plot.png")


if __name__ == "__main__":
    out_dir = Path("./doodles")
    out_dir.mkdir(exist_ok=True, parents=True)
    generate("ER", 50, 3, out_dir)
    generate("PA", 50, 3, out_dir)
