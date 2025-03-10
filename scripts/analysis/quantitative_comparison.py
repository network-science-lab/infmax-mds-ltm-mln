"""A script to produce comparison when mds was better both in terms of gain and dynamics."""

import glob
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from src.aux import auc, slicer_plotter


def load_data() -> slicer_plotter.ResultsSlicer:
    print("loading data")
    return slicer_plotter.ResultsSlicer(
        [
            csv_file for csv_file in glob.glob(r"data/raw_results/**", recursive=True)
            if re.search(r"batch_([1-9][0-2]?)/.*\.csv$", csv_file)
        ]
    )

def produce_quantitative_results(
    results: slicer_plotter.ResultsSlicer,
    mds_type: Literal["d^", "D^"]
) -> pd.DataFrame:
    quantitative_results_aggr = []

    for page_case in slicer_plotter.ResultsPlotter().yield_page():
        print(page_case)

        for fig_case in slicer_plotter.ResultsPlotter().yield_figure(protocol=page_case[1]):
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

            cdf_dict = auc.prepare_cdfs(mds_slice=mds_slice, nml_slice=nml_slice)
            try:
                mds_auc = auc.area_under_curve(
                    cdf=cdf_dict["mds_cdf"],
                    start_val=cdf_dict["start_val"],
                    max_value=cdf_dict["max_val"],
                )
            except ValueError:
                mds_auc = 0.
            try:
                nml_auc = auc.area_under_curve(
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
    root_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = root_dir / Path("./data/processed_results/")
    out_dir.mkdir(exist_ok=True, parents=True)
    raw_results = load_data()
    quantitative_results_df = produce_quantitative_results(raw_results, "D^")
    quantitative_results_df.to_csv(out_dir / "quantitative_comparison.csv")
