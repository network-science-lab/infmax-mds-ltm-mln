"""A script to produce comparison when mds was better both in terms of gain and dynamics."""

import glob
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src import visualisation


def prepare_cdfs(mds_slice: pd.DataFrame, nml_slice: pd.DataFrame) -> dict[str, np.ndarray | int]:
    # compute raw cdf
    mds_cdf = visualisation.Results.mean_expositions_rec(mds_slice)["cdf"]
    nml_cdf = visualisation.Results.mean_expositions_rec(nml_slice)["cdf"]
    # pad the shorter one
    padding_size = max(len(nml_cdf), len(mds_cdf))
    nml_cdf = np.pad(
        nml_cdf,
        (0, np.abs(padding_size - len(nml_cdf))),
        "constant",
        constant_values=nml_cdf[-1],
    )
    mds_cdf = np.pad(
        mds_cdf,
        (0, np.abs(padding_size - len(mds_cdf))),
        "constant",
        constant_values=mds_cdf[-1],
    )
    # obtain max available value and discounting number
    nb_actors = (mds_slice["exposed_nb"] + mds_slice["unexposed_nb"]).iloc[0].item()
    seed_nb = nml_slice["seed_nb"].iloc[0].item()
    # return CDFs and min/max values
    return {
        "mds_cdf": mds_cdf,
        "nml_cdf": nml_cdf,
        "max_val": nb_actors,
        "start_val": seed_nb,
    }


def area_under_curve(cdf: np.ndarray, start_val: int, max_value: int) -> float:
    if len(cdf) < 2:
        raise ValueError("CDF must contain at least two values.")
    area = np.trapezoid(cdf - start_val)
    return area / ((max_value - start_val) * len(cdf))


def load_data(batches: str = "1,2,3,4") -> visualisation.Results:
    print("loading data")
    return visualisation.Results(
        [
            csv_file for csv_file in glob.glob(r"data/raw_results/**", recursive=True)
            if re.search(fr"batch_([{batches}])/.*\.csv$", csv_file)
        ]
    )

def produce_quantitative_results(
    results: visualisation.Results,
    mds_type: Literal["d^", "D^"]
) -> pd.DataFrame:
    quantitative_results_aggr = []

    for page_case in visualisation.Plotter().yield_page():
        print(page_case)

        for fig_case in visualisation.Plotter().yield_figure(protocol=page_case[1]):
            print(f"\t{fig_case}")

            nml_slice = results.get_slice(
                protocol=page_case[1],
                mi_value=fig_case[1],
                seed_budget=fig_case[0],
                network=page_case[0],
                ss_method=page_case[2],
            )
            mds_slice = results.get_slice(
                protocol=page_case[1],
                mi_value=fig_case[1],
                seed_budget=fig_case[0],
                network=page_case[0],
                ss_method=f"{mds_type}{page_case[2]}",
            )

            if len(nml_slice) == 0 or len(mds_slice) == 0:
                continue

            mds_gain = mds_slice["gain"].mean()
            nml_gain = nml_slice["gain"].mean()

            cdf_dict = prepare_cdfs(mds_slice=mds_slice, nml_slice=nml_slice)
            try:
                mds_auc = area_under_curve(
                    cdf=cdf_dict["mds_cdf"],
                    start_val=cdf_dict["start_val"],
                    max_value=cdf_dict["max_val"],
                )
            except ValueError:
                mds_auc = 0.
            try:
                nml_auc = area_under_curve(
                    cdf=cdf_dict["nml_cdf"],
                    start_val=cdf_dict["start_val"],
                    max_value=cdf_dict["max_val"],
                )
            except ValueError:
                nml_auc = 0.

            quantitative_results_aggr.append(
                {
                    "protocol": page_case[1],
                    "mi_value": fig_case[1],
                    "seed_budget": fig_case[0],
                    "network": page_case[0],
                    "ss_method": page_case[2],
                    "mds_gain": mds_gain,
                    "mds_auc": mds_auc,
                    "nml_gain": nml_gain,
                    "nml_auc": nml_auc,
                    "gain_winner": "mds" if mds_gain > nml_gain else "nml",
                    "auc_winner": "mds" if mds_auc > nml_auc else "nml",
                }
            )
    return pd.DataFrame(quantitative_results_aggr)


if __name__ == "__main__":
    out_dir = Path("./data/processed_results/")
    out_dir.mkdir(exist_ok=True, parents=True)
    raw_results = load_data("1,2,3,4")
    quantitative_results_df = produce_quantitative_results(raw_results, "d^")
    quantitative_results_df.to_csv(out_dir / "quantitative_comparison.csv")
