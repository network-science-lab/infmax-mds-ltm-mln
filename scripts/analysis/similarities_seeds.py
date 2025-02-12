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


def squash_sets(all_sets: list[set[str]]) -> str:
    unique_ss = set()
    for ss in all_sets:
        unique_ss = unique_ss.union(ss)
    return unique_ss


def get_seed_sets(
    raw_df: pd.DataFrame,
    network: str,
    seed_budget: int,
    ss_method: str,
    protocol: str,
    repetition: str,
) -> list[set[str]]:
    seed_sets = raw_df.loc[
        (raw_df["network"] == network) &
        (raw_df["seed_budget"] == seed_budget) &
        (raw_df["ss_method"] == ss_method) &
        (raw_df["protocol"] == protocol) &
        (raw_df["repetition"] == repetition) &
        (raw_df["mi_value"] == 0.1)  # it doesn't matter for the results but speeds up computations
    ]["seed_ids"].to_list()
    return [set(seed_set.split(";")) for seed_set in seed_sets]  # a workaround if seed set is empty


def raw_df_to_table(raw_df: pd.DataFrame, net_type: str, metric: str) -> pd.DataFrame:
    filtered_df = raw_df.loc[raw_df["type"] == net_type].drop(
        ["type", "network", "protocol", "repetition"], axis=1
    )
    avg_df = filtered_df.groupby(["seed_budget", "ss_method"]).mean().reset_index()
    avg_pivot_df = pd.pivot_table(avg_df, index="seed_budget", columns="ss_method", values=metric)
    avg_pivot_df = avg_pivot_df.map(lambda x: f"{x:.2f}")
    std_df = filtered_df.groupby(["seed_budget", "ss_method"]).std().reset_index()
    std_pivot_df = pd.pivot_table(std_df, index="seed_budget", columns="ss_method", values=metric)
    std_pivot_df = std_pivot_df.map(lambda x: f"{x:.0E}")
    return "$" + avg_pivot_df + " (" + std_pivot_df + ")$"


def similarities_seeds(results: ResultsSlicer, net_type: str) -> pd.DataFrame:
    print(f"plotting statistics about seed sets to the manuscript: {net_type}")
    iterator = product(
        ResultsPlotter._networks,
        set(ResultsPlotter._seed_budgets_and).union(set(ResultsPlotter._seed_budgets_or)),
        ResultsPlotter._ss_methods,
        [ResultsPlotter._protocol_or, ResultsPlotter._protocol_and],
        sorted(results.raw_df["repetition"].unique().tolist()),
    )
    iterator = list(iterator)
    similarity_list = []
    for simulated_case in tqdm(iterator):
        network = simulated_case[0]
        seed_budget = simulated_case[1]
        ss_method = simulated_case[2]
        protocol = simulated_case[3]
        repetition = simulated_case[4]
        if ResultsPlotter._networks_groups[network] != net_type:
            continue

        mds_seed_sets = get_seed_sets(
            raw_df=results.raw_df,
            network=network,
            seed_budget=seed_budget,
            ss_method=f"D^{ss_method}",
            protocol = protocol,
            repetition=repetition,
        )
        if len(mds_seed_sets) == 0:
            continue

        nml_seed_sets = get_seed_sets(
            raw_df=results.raw_df,
            network=network,
            seed_budget=seed_budget,
            ss_method=ss_method,
            protocol = protocol,
            repetition=repetition,
        )

        intersection = nml_seed_sets[0].intersection(mds_seed_sets[0])
        union =  nml_seed_sets[0].union(mds_seed_sets[0])

        similarity_list.append(
            {
                "network": network,
                "type": ResultsPlotter._networks_groups[network],
                "seed_budget": seed_budget,
                "ss_method": ss_method,
                "protocol": protocol,
                "repetition": repetition,
                "jaccard": len(intersection) / len(union),
            }
        )

    similarity_df = pd.DataFrame(similarity_list)

    table = raw_df_to_table(similarity_df, net_type, "jaccard")
    table.to_latex(f"{net_type}_jaccard.txt")


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
        ],
        with_repetition=True,
    )

    # compute the DFs
    seeds_similarity_and_df = generate_similarities_seeds(results, "AND")
    seeds_similarity_and_df.to_csv(workdir.joinpath("similarities_seeds_and.csv"))
    seeds_similarity_or_df = generate_similarities_seeds(results, "OR")
    seeds_similarity_or_df.to_csv(workdir.joinpath("similarities_seeds_or.csv"))

    # this goes to publication
    similarities_seeds(results, "real")
    similarities_seeds(results, "er")
    similarities_seeds(results, "sf")