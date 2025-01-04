"""A script to produce html files with quantitative analysis of experiments where MDS won."""

from pathlib import Path
from typing import Literal
import pandas as pd
import ydata_profiling


def main(
    quantitative_comparison_path: Path,
    out_dir: Path,
    split_type: Literal["gain", "auc", "gain_and_auc", "gain_or_auc"]
) -> None:
    # read csv with quantitative comparison
    df_raw = pd.read_csv(quantitative_comparison_path)

    # scale doen gain to be in the same range as auc
    df_raw["mds_gain"] = df_raw["mds_gain"] / 100
    df_raw["nml_gain"] = df_raw["nml_gain"] / 100

    # drop records where there was no diffusion and those where random was used as ssm
    df = df_raw.drop(df_raw.loc[
        (df_raw["mds_gain"] == 0.) & (df_raw["nml_gain"] == .0)
    ].index).reset_index()
    df = df.drop(df.loc[df["ss_method"] == "random"].index).reset_index()

    # case 1: get those cases where MDS attained better gain 
    if split_type == "gain":
        ddf = df.loc[(df["mds_gain"] - df["nml_gain"] >= .01)]

    # case 2: get those cases where MDS attained better auc 
    elif split_type == "auc":
        ddf = df.loc[(df["mds_auc"] - df["nml_auc"] >= .01)]

    # case 3: get those cases where MDS attained better gain and also better auc
    elif split_type == "gain_and_auc":
        ddf = df.loc[
            (df["mds_gain"] - df["nml_gain"] >= .01) &
            (df["mds_auc"] - df["nml_auc"] >= .01)
        ]

    # case 4: get those cases where MDS attained either better gain or better auc
    elif split_type == "gain_or_auc":
        ddf = df.loc[
            (df["mds_gain"] - df["nml_gain"] >= .01) |
            (df["mds_auc"] - df["nml_auc"] >= .01)
        ]
    
    else:
        raise ValueError("unknown split type!")

    print(f"Current split takes {round(len(ddf) / len(df) * 100, 3)} prct of valid experiments")
    report = ydata_profiling.ProfileReport(ddf)
    report.to_file(out_dir / f"{split_type}.html")


if __name__ == "__main__":
    in_path = Path("data/processed_results/quantitative_comparison.csv")
    out_dir = Path("data/processed_results/profile_reports")
    main(in_path, out_dir, "gain")
    main(in_path, out_dir, "auc")
    main(in_path, out_dir, "gain_and_auc")
    main(in_path, out_dir, "gain_or_auc")
