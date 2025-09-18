"""A pipeline to plot MDS on random and real networks"""

from pathlib import Path

from src.aux.network_generator import generate
from src.aux.visualise_mds import MDSVisualiser
from src.models.mds import greedy_search, local_improvement
from src.loaders.net_loader import load_network
from src.aux.slicer_plotter import ResultsPlotter


if __name__ == "__main__":

    mds_func = local_improvement.get_mds_locimpr
    # mds_func = greedy_search.get_mds_greedy

    # prepare outout directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = root_dir / "data/processed_results/visualisations_mds"
    out_dir.mkdir(exist_ok=True, parents=True)
    print(out_dir)

    # visualise artificial networks
    n_networks = 5
    for idx in range(n_networks):
        print(f"{idx + 1}/{n_networks}")
        # Erdos-Renyi model
        net = generate(model="ER", nb_actors=35, nb_layers=2)
        mds = mds_func(net)
        mds_plotter = MDSVisualiser(net, mds, f"ER_{idx + 1}", out_dir)
        mds_plotter.manuscript_plot()
        # Preferential-Attachment model
        net = generate(model="PA", nb_actors=35, nb_layers=2)
        mds = mds_func(net)
        mds_plotter = MDSVisualiser(net, mds, f"PA_{idx + 1}", out_dir)
        mds_plotter.manuscript_plot()

    # # visualise real networks
    # for net_name in ResultsPlotter._networks:
    #     if net_name == "timik1q2009":
    #         continue
    #     print(net_name)
    #     net = load_network(net_name)
    #     mds = mds_func(net)
    #     mds_plotter = MDSVisualiser(net, mds, net_name, out_dir)
    #     mds_plotter.plot_centralities()
    #     mds_plotter.plot_structure()
