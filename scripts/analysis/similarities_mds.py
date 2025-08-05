"""Statistical analysis of MDS rankings."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.aux import slicer_plotter
from src.loaders.net_loader import load_network


def generate_similarities_mds() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("plotting statistics about MDS used")
    zip_1_path = "data/raw_results/batch_7/rankings.zip"
    zip_2_path = "data/raw_results/batch_8/rankings.zip"
    zip_3_path = "data/raw_results/batch_9/rankings.zip"
    zip_4_path = "data/raw_results/batch_10/rankings.zip"
    zip_5_path = "data/raw_results/batch_11/rankings.zip"
    zip_6_path = "data/raw_results/batch_12/rankings.zip"
    zip_7_path = "data/raw_results/batch_15/rankings.zip"
    zip_8_path = "data/raw_results/batch_16/rankings.zip"
    used_mds_list = [
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_1_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_2_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_3_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_4_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_5_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_6_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_7_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_8_path),
    ]
    used_mds_df = pd.DataFrame(used_mds_list)

    iterator_mds = product(
        # used_mds_df["ss_method"].unique(),
        used_mds_df["network"].unique(),
    )
    iterator_mds = list(iterator_mds)

    mds_similarity_list = []
    for idx, simulated_case in enumerate(tqdm(iterator_mds)):
        case_mds = used_mds_df.loc[
            # (used_mds_df["ss_method"] == simulated_case[0]) &
            (used_mds_df["network"] == simulated_case[0])
        ]["mds"]
        mds_lengths = [len(cm) for cm in case_mds]
        mds_similarity = slicer_plotter.analyse_set_similarity(case_mds)
        mds_similarity_list.append(
            {
                "network": simulated_case[0],
                # "ss_method": simulated_case[0],
                "max_mds_length": np.max(mds_lengths),
                "min_mds_length": np.min(mds_lengths),            
                "avg_mds_length": np.mean(mds_lengths),
                "std_mds_length": np.std(mds_lengths),
                **mds_similarity,
            }
        )

    mds_similarity_df = pd.DataFrame(mds_similarity_list)
    return mds_similarity_df, used_mds_df


def similarities_mds_to_latex(csv_path: Path) -> None:
    print("Creating manuscript-ready table.")

    # read the data and add network size
    actors_nbs = {}
    ms_df = pd.read_csv(csv_path, index_col=0)
    for net_name in ms_df["network"]:
        net_graph = load_network(net_name)
        actors_nbs[net_name] = net_graph.get_actors_num()
    ms_df = ms_df.set_index("network")
    ms_df.loc[:, "net_size"] = actors_nbs

    # create a colum for the normalised MDS-size range
    ms_df["max_mds_length"] /= ms_df["net_size"]
    ms_df["min_mds_length"] /= ms_df["net_size"]
    ms_df["size_range"] = (
        "$[" +
        ms_df["min_mds_length"].apply(lambda x: f"{x:.2f}") + 
        ", " +
        ms_df["max_mds_length"].apply(lambda x: f"{x:.2f}") +
        "]$"
    )

    # create a colum for the normalised MDS-average range
    ms_df["avg_mds_length"] /= ms_df["net_size"]
    ms_df["std_mds_length"] /= ms_df["net_size"]
    ms_df["avg_size"] = (
        "$" +
        ms_df["avg_mds_length"].apply(lambda x: f"{x:.2f}") +  
        " (" +
        # ms_df["std_mds_length"].round(4).astype(str) +
        ms_df["std_mds_length"].apply(lambda x: f"{x:.0E}") +
        ")$"
    )

    # convert other columns to math symbols
    ms_df["entropy"] = "$" + ms_df["entropy"].apply(lambda x: f"{x:.2f}") + "$"
    ms_df["jaccard_similarity"] = "$" + ms_df["jaccard_similarity"].apply(lambda x: f"{x:.2f}") + "$"
    ms_df["unique_sets_ratio"] = "$" + ms_df["unique_sets_ratio"] + "$"

    # save to latex
    ms_df[
        [
            "size_range",
            "avg_size",
            "unique_sets_ratio",
            "jaccard_similarity",
            "entropy",
        ]
    ].reset_index().to_latex("table.txt")


if __name__ == "__main__":

    # prepare outout directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    workdir = root_dir / "data/processed_results"
    workdir.mkdir(exist_ok=True, parents=True)

    # compute the DFs
    mds_similarity_df, used_mds_df = generate_similarities_mds()
    mds_similarity_df.to_csv(workdir.joinpath("similarities_mds.csv"))
    used_mds_df.to_csv(workdir.joinpath("used_mds.csv"))

    # prepare a publication-ready table
    similarities_mds_to_latex(workdir.joinpath("similarities_mds.csv"))
