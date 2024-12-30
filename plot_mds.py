"""A pipeline to plot MDS on random and real networks"""

from pathlib import Path

from src.aux.network_generator import generate
from src.aux.visualise_mds import MDSPlotter
from src.models.mds import greedy_search, local_improvement
from src.loaders.net_loader import load_network


if __name__ == "__main__":

    mds_func = local_improvement.get_mds_locimpr
    mds_func = greedy_search.get_mds_greedy

    # prepare outout directory
    out_dir = Path("./doodles")
    out_dir.mkdir(exist_ok=True, parents=True)

    # visualise artificial networks
    for idx in range(2):
        # Erdos-Renyi model
        net = generate(model="ER", nb_actors=50, nb_layers=3)
        mds = mds_func(net)
        mds_plotter = MDSPlotter(net, mds, f"ER_{idx}", out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()
        # Preferential-Attachment model
        net = generate(model="PA", nb_actors=100, nb_layers=3)
        mds = mds_func(net)
        mds_plotter = MDSPlotter(net, mds, f"PA_{idx}", out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()

    # visualise real networks
    for net_name in ["aucs", "l2_course_net_1"]: # Plotter._networks:
        net = load_network(net_name, as_tensor=False)
        mds = mds_func(net)
        mds_plotter = MDSPlotter(net, mds, net_name, out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()
