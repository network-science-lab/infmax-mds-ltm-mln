
import glob
import re
from typing import Literal

import matplotlib
from src.aux import MDS_ACTORS_COLOUR, NML_ACTORS_COLOUR, auc, slicer_plotter
from src.result_handler import compute_gain

import matplotlib.pyplot as plt
import numpy as np


case_mi_value = 0.4
case_seed_budget = 35
case_network = "er3"
case_protocol = "AND"
case_ss_method = "nghb_sd"


if __name__ == "__main__":
    raw_results = slicer_plotter.ResultsSlicer(
        [
            csv_file for csv_file in glob.glob(r"data/raw_results/**", recursive=True)
            if re.search(r"batch_([3,9])/.*\.csv$", csv_file)
        ]
    )
    nml_slice = raw_results.get_slice(
        protocol=case_protocol,
        mi_value=case_mi_value,
        seed_budget=case_seed_budget,
        network=case_network,
        ss_method=case_ss_method,
    )
    mds_slice = raw_results.get_slice(
        protocol=case_protocol,
        mi_value=case_mi_value,
        seed_budget=case_seed_budget,
        network=case_network,
        ss_method=f"D^{case_ss_method}",
    )

    cdfs_slice = auc.prepare_cdfs(mds_slice=mds_slice, nml_slice=nml_slice)
    auc_mds = auc.area_under_curve(cdfs_slice["mds_cdf"], cdfs_slice["start_val"], cdfs_slice["max_val"])
    auc_nml = auc.area_under_curve(cdfs_slice["nml_cdf"], cdfs_slice["start_val"], cdfs_slice["max_val"])
    gain_mds = compute_gain(cdfs_slice["start_val"], max(cdfs_slice["mds_cdf"]), cdfs_slice["max_val"])
    gain_nml = compute_gain(cdfs_slice["start_val"], max(cdfs_slice["nml_cdf"]), cdfs_slice["max_val"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    x = np.arange(0, len(cdfs_slice["mds_cdf"]))
    ax.hlines(y=cdfs_slice["start_val"], xmin=min(x), xmax=max(x), color="red")
    ax.hlines(y=cdfs_slice["max_val"], xmin=min(x), xmax=max(x), color="red")
    ax.plot(
        x,
        cdfs_slice["nml_cdf"],
        color=NML_ACTORS_COLOUR,
        label=f"baseline: Γ={round(gain_nml / 100, 2)}, Λ={round(auc_nml, 2)}",
    )
    ax.fill_between(
        x,
        cdfs_slice["nml_cdf"],
        [cdfs_slice["start_val"]] * len(x),
        facecolor=NML_ACTORS_COLOUR,
        edgecolor=NML_ACTORS_COLOUR,
        alpha=0.25,
        hatch="||",
    )
    ax.plot(
        x,
        cdfs_slice["mds_cdf"],
        color=MDS_ACTORS_COLOUR,
        label=f"mds-filtered: Γ={round(gain_mds / 100, 2)}, Λ={round(auc_mds, 2)}",
    )
    ax.fill_between(
        x,
        cdfs_slice["mds_cdf"],
        [cdfs_slice["start_val"]] * len(x),
        facecolor=MDS_ACTORS_COLOUR,
        edgecolor=MDS_ACTORS_COLOUR,
        alpha=0.25,
        hatch="\\\\",
    )

    ax.set_xlim(left=min(x), right=max(x))
    ax.set_ylim(bottom=0, top=1.1 * cdfs_slice["max_val"])
    ax.legend(loc="lower right")

    ax_aux_yticks_locations = np.linspace(cdfs_slice["start_val"], cdfs_slice["max_val"], 5)

    ax_x_aux = ax.twinx()
    ax_x_aux.set_ylim(ax.get_ylim())
    ax_x_aux.set_yticks(ax_aux_yticks_locations)
    ax_x_aux.set_yticklabels(
        (ax_aux_yticks_locations - cdfs_slice["start_val"]) /
        (cdfs_slice["max_val"] - cdfs_slice["start_val"])
    )
    ax_x_aux.tick_params(axis="y", colors="red")

    ax_y_aux = ax.twiny()
    ax_y_aux.set_xlim(ax.get_xlim())
    ax_y_aux.set_xticks(ax.get_xticks())
    ax_y_aux.set_xticklabels(ax.get_xticks() / max(ax.get_xticks()))
    ax_y_aux.tick_params(axis="x", colors="red")

    fig.tight_layout()
    fig.savefig(f"data/processed_results/metrics.pdf")
