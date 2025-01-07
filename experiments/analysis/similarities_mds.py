"""Statistical analysis of MDS rankings."""

import sys
from itertools import product
from pathlib import Path

root_path = Path(".").resolve()
sys.path.append(str(root_path))

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.aux import slicer_plotter

def generate_similarities_mds() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("plotting statistics about MDS used")
    zip_1_path = "data/raw_results/batch_7/rankings.zip"
    zip_2_path = "data/raw_results/batch_8/rankings.zip"
    zip_3_path = "data/raw_results/batch_9/rankings.zip"
    zip_4_path = "data/raw_results/batch_10/rankings.zip"
    zip_5_path = "data/raw_results/batch_11/rankings.zip"
    zip_6_path = "data/raw_results/batch_12/rankings.zip"
    used_mds_list = [
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_1_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_2_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_3_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_4_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_5_path),
        *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_6_path),
    ]
    used_mds_df = pd.DataFrame(used_mds_list)
    used_mds_df

    iterator_mds = product(
        used_mds_df["ss_method"].unique(),
        used_mds_df["network"].unique(),
    )
    iterator_mds = list(iterator_mds)

    mds_similarity_list = []
    for idx, simulated_case in enumerate(tqdm(iterator_mds)):
        case_mds = used_mds_df.loc[
            (used_mds_df["ss_method"] == simulated_case[0]) &
            (used_mds_df["network"] == simulated_case[1])
        ]["mds"]
        mds_lengths = [len(cm) for cm in case_mds]
        mds_similarity = slicer_plotter.analyse_set_similarity(case_mds)
        mds_similarity_list.append(
            {
                "network": simulated_case[1],
                "ss_method": simulated_case[0],
                "max_mds_length": np.max(mds_lengths),
                "min_mds_length": np.min(mds_lengths),            
                "avg_mds_length": np.mean(mds_lengths),
                "std_mds_length": np.std(mds_lengths),
                **mds_similarity,
            }
        )

    mds_similarity_df = pd.DataFrame(mds_similarity_list)
    return mds_similarity_df, used_mds_df


if __name__ == "__main__":

    # prepare outout directory
    workdir = root_path / "data/processed_results"
    workdir.mkdir(exist_ok=True, parents=True)

    # compute the DFs
    mds_similarity_df, used_mds_df = generate_similarities_mds()
    mds_similarity_df.to_csv(workdir.joinpath("similarities_mds.csv"))
    used_mds_df.to_csv(workdir.joinpath("used_mds.csv"))
