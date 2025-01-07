"""A pipeline to plot MDS on random and real networks"""

import sys
from pathlib import Path

root_path = Path(".").resolve()
sys.path.append(str(root_path))

from src.aux.network_generator import generate
from src.aux.visualise_mds import MDSVisualiser
from src.models.mds import greedy_search, local_improvement
from src.loaders.net_loader import load_network
from src.aux.slicer_plotter import ResultsPlotter


if __name__ == "__main__":

    mds_func = local_improvement.get_mds_locimpr
    # mds_func = greedy_search.get_mds_greedy

    # prepare outout directory
    out_dir = root_path / "data/processed_results/visualisations_mds"
    out_dir.mkdir(exist_ok=True, parents=True)
    print(out_dir)

    # visualise artificial networks
    for idx in range(10):
        print(f"{idx + 1}/{10}")
        # Erdos-Renyi model
        net = generate(model="ER", nb_actors=50, nb_layers=3)
        mds = mds_func(net)
        mds_plotter = MDSVisualiser(net, mds, f"ER_{idx + 1}", out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()
        # Preferential-Attachment model
        net = generate(model="PA", nb_actors=100, nb_layers=3)
        mds = mds_func(net)
        mds_plotter = MDSVisualiser(net, mds, f"PA_{idx + 1}", out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()

    # visualise real networks
    for net_name in ResultsPlotter._networks:
        print(net_name)
        net = load_network(net_name, as_tensor=False)
        mds = mds_func(net)
        mds_plotter = MDSVisualiser(net, mds, net_name, out_dir)
        mds_plotter.plot_centralities()
        mds_plotter.plot_structure()
