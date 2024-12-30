"""A pipeline to generate and plot random networks with MDS."""

from pathlib import Path
from typing import Literal

import network_diffusion as nd
import uunet.multinet as ml

from src.network_generator import MultilayerERGenerator, MultilayerPAGenerator, MDSPlotter
from src.models.mds import greedy_search, local_improvement
from src.loaders.net_loader import load_network


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
    actors_to_remove = [ac.actor_id for ac, nghb in nd.mln.functions.neighbourhood_size(net_nd).items() if nghb == 0]
    for l_graph in net_nd.layers.values():
        l_graph.remove_nodes_from(actors_to_remove)
    print(net_nd)
    return net_nd


if __name__ == "__main__":

    out_dir = Path("./doodles")
    out_dir.mkdir(exist_ok=True, parents=True)

    # for idx in range(10):
    #     net = generate("ER", 50, 3)
    #     mds = local_improvement.get_mds_locimpr(net)
    #     plot(net, mds, f"ER_{idx}", out_dir)

    #     net = generate("PA", 100, 3)
    #     mds = local_improvement.get_mds_locimpr(net)
    #     plot(net, mds, f"PA_{idx}", out_dir)

    for net_name in ["aucs", "l2_course_net_1"]: # Plotter._networks:
        net = load_network(net_name, as_tensor=False)
        mds = local_improvement.get_mds_locimpr(net)
        plotter = MDSPlotter(net, mds, net_name, out_dir)
        plotter.plot_centralities()
        plotter.plot_structure()
    

    # net = generate("ER", 50, 3)
    # mds = local_improvement.get_mds_locimpr(net)
    # plot(net, mds, "ER", out_dir)

    # net = generate("PA", 100, 3)
    # mds = local_improvement.get_mds_locimpr(net)
    # plot(net, mds, "PA", out_dir)
