"""Statistical analysis of MDS rankings."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.aux import slicer_plotter
from src.loaders.net_loader import load_network


def generate_similarities_mds() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("plotting statistics on differences between MDS obtained with greedy and local iprov.")
    # zip_1_path = "data/raw_results/batch_1/rankings.zip"
    # zip_2_path = "data/raw_results/batch_2/rankings.zip"
    # zip_3_path = "data/raw_results/batch_3/rankings.zip"
    # zip_4_path = "data/raw_results/batch_4/rankings.zip"
    # zip_5_path = "data/raw_results/batch_5/rankings.zip"
    # zip_6_path = "data/raw_results/batch_6/rankings.zip"
    # zip_7_path = "data/raw_results/batch_7/rankings.zip"
    # zip_8_path = "data/raw_results/batch_8/rankings.zip"
    # zip_9_path = "data/raw_results/batch_9/rankings.zip"
    # zip_10_path = "data/raw_results/batch_10/rankings.zip"
    # zip_11_path = "data/raw_results/batch_11/rankings.zip"
    # zip_12_path = "data/raw_results/batch_12/rankings.zip"
    # used_mds_list = [
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_1_path),
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_2_path),
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_3_path),
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_4_path),
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_5_path),
    #     *slicer_plotter.JSONParser().read_minimal_dominating_sets(zip_6_path),
    # ]

    batches = [
        "batch_1",
        "batch_2",
        "batch_3",
        "batch_4",
        "batch_5",
        "batch_6",
        "batch_7",
        "batch_8",
        "batch_9",
        "batch_10",
        "batch_11",
        "batch_12",
    ]
    used_mds_list = []
    for batch_id in batches:
        print(batch_id)
        used_mds_list.extend(
            slicer_plotter.JSONParser().read_minimal_dominating_sets(
                f"data/raw_results/{batch_id}/rankings.zip"
            )
        )

    used_mds_df = pd.DataFrame(used_mds_list)
    print(used_mds_df)

    # iterator_mds = product(
    #     # used_mds_df["ss_method"].unique(),
    #     used_mds_df["network"].unique(),
    # )
    # iterator_mds = list(iterator_mds)

    # mds_similarity_list = []
    # for idx, simulated_case in enumerate(tqdm(iterator_mds)):
    #     case_mds = used_mds_df.loc[
    #         # (used_mds_df["ss_method"] == simulated_case[0]) &
    #         (used_mds_df["network"] == simulated_case[0])
    #     ]["mds"]
    #     mds_lengths = [len(cm) for cm in case_mds]
    #     mds_similarity = slicer_plotter.analyse_set_similarity(case_mds)
    #     mds_similarity_list.append(
    #         {
    #             "network": simulated_case[0],
    #             # "ss_method": simulated_case[0],
    #             "max_mds_length": np.max(mds_lengths),
    #             "min_mds_length": np.min(mds_lengths),            
    #             "avg_mds_length": np.mean(mds_lengths),
    #             "std_mds_length": np.std(mds_lengths),
    #             **mds_similarity,
    #         }
    #     )

    # mds_similarity_df = pd.DataFrame(mds_similarity_list)
    # return mds_similarity_df, used_mds_df


if __name__ == "__main__":

    # prepare outout directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    workdir = root_dir / "data/processed_results"
    workdir.mkdir(exist_ok=True, parents=True)

    # compute the DFs
    mds_similarity_df, used_mds_df = generate_similarities_mds()
    # mds_similarity_df.to_csv(workdir.joinpath("_f_similarities_mds.csv"))
    # used_mds_df.to_csv(workdir.joinpath("used_mds.csv"))
