"""Statistical analysis of seed sets used in experiments."""

import glob
import re
from itertools import product
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm
from src.aux.slicer_plotter import ResultsPlotter, ResultsSlicer, analyse_set_similarity

def generate_similarities_seeds(
    results: ResultsSlicer, protocol: Literal["AND", "OR"]
) -> pd.DataFrame:
    print("plotting statistics about seed sets")
    iterator = product(
        ResultsPlotter._networks,
        [protocol],
        ResultsPlotter._seed_budgets_and if protocol == "AND" else ResultsPlotter._seed_budgets_or,
        ResultsPlotter._mi_values,
        [*[f"D^{ssm}" for ssm in ResultsPlotter._ss_methods], *ResultsPlotter._ss_methods]
    )
    iterator = list(iterator)

    similarity_list = []

    for simulated_case in tqdm(iterator):
        seed_sets = results.obtain_seed_sets_for_simulated_case(results.raw_df, *simulated_case)
        similarity = analyse_set_similarity(seed_sets)
        similarity_list.append(
            {
                "network": simulated_case[0],
                "protocol": simulated_case[1],
                "seed_budget": simulated_case[2],
                "mi_value": simulated_case[3],
                "ss_method": simulated_case[4],
                **similarity,
            }
        )

    similarity_df = pd.DataFrame(similarity_list)
    return similarity_df


if __name__ == "__main__":

    # prepare outout directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    workdir = root_dir / "data/processed_results"
    workdir.mkdir(exist_ok=True, parents=True)

    # read raw results
    results = ResultsSlicer(
        [
            csv_file for csv_file in glob.glob(fr"{str(root_dir)}/data/raw_results/**", recursive=True)
            if re.search(r"batch_([1-9][0-2]?)/.*\.csv$", csv_file)
        ]
    )

    # compute the DFs
    seeds_similarity_and_df = generate_similarities_seeds(results, "AND")
    seeds_similarity_and_df.to_csv(workdir.joinpath("similarities_seeds_and.csv"))
    seeds_similarity_or_df = generate_similarities_seeds(results, "OR")
    seeds_similarity_or_df.to_csv(workdir.joinpath("similarities_seeds_or.csv"))
