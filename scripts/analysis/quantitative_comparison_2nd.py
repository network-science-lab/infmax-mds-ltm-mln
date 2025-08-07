"""A script to produce comparison when mds was better both in terms of gain and dynamics."""

import glob
import re
from itertools import product
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from src.aux import auc


class ResultsSlicer2:

    def __init__(self, raw_results_path: list[str], with_repetition: bool = False) -> None:
        self.raw_df = self.read_raw_df(raw_results_path, with_repetition)

    @staticmethod
    def read_raw_df(raw_result_paths: list[str], with_repetition: bool) -> pd.DataFrame:
        dfs = []
        print(set(str(Path(csv_path).parent) for csv_path in raw_result_paths))
        for csv_path in raw_result_paths:
            csv_df = pd.read_csv(csv_path)
            if with_repetition:
                csv_df["repetition"] = Path(csv_path).stem.split("_")[-1]
            dfs.append(csv_df)
        concat_df = pd.concat(dfs, axis=0, ignore_index=True)
        concat_df[["group", "series", "repetition"]] = (
            concat_df["network"]
            .apply(lambda x: x.split("-")[-3:])
            .apply(pd.Series)
            .rename(columns={0: "group", 1: "series", 2: "repetition"})
        )
        return concat_df

    def get_slice(
        self,
        protocol: str,
        mi_value: float,
        seed_budget: int,
        ss_method: str,
        group: str,
        series: str,
    ) -> pd.DataFrame:
        slice_df = self.raw_df.loc[
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["ss_method"] == ss_method) &
            (self.raw_df["group"] == group) &
            (self.raw_df["series"] == series)
        ].copy()
        slice_df["expositions_rec"] = slice_df["expositions_rec"].map(lambda x: [int(xx) for xx in x.split(";")])
        return slice_df.reindex()

    @staticmethod
    def cdf_expositions(slice_df: pd.DataFrame, padded_length: int) -> tuple[np.ndarray, np.ndarray]:
        exp_recs_padded = np.zeros([len(slice_df), padded_length])
        max_available_values = np.zeros([len(slice_df), 1])
        for run_idx, (_, row) in enumerate(slice_df.iterrows()):
            exp_recs_padded[run_idx][0:len(row["expositions_rec"])] = row["expositions_rec"]
            max_available_values[run_idx] = row["exposed_nb"] + row["unexposed_nb"]
        return np.cumsum(exp_recs_padded, axis=1), max_available_values
    
    @staticmethod
    def compute_auc(mds_slice: pd.DataFrame, nml_slice: pd.DataFrame) -> dict[str, float]:
        
        # find the padding length
        padded_len = max(
            max(mds_slice["expositions_rec"].apply(lambda x: len(x))),
            max(nml_slice["expositions_rec"].apply(lambda x: len(x)))
        )

        # compute CDF and then AUC for MDS-backed slice
        mds_cdfs, mds_maxs = ResultsSlicer2.cdf_expositions(mds_slice, padded_len)
        mds_aucs = np.apply_along_axis(
            lambda x: auc.area_under_curve(x[:-1], x[0], x[-1]),
            axis=1,
            arr=np.append(mds_cdfs, mds_maxs, axis=1), # append nb actors to the CDF arr
        )

        # compute CDF and then AUC for baseline slice
        nml_cdfs, nml_maxs = ResultsSlicer2.cdf_expositions(nml_slice, padded_len)
        nml_aucs = np.apply_along_axis(
            lambda x: auc.area_under_curve(x[:-1], x[0], x[-1]),
            axis=1,
            arr=np.append(nml_cdfs, nml_maxs, axis=1),  # append nb actors to the CDF arr
        )

        # return avg and std of AUCs values
        return {
            "mds_mean": mds_aucs.mean(),
            "mds_std": mds_aucs.std(),
            "nml_mean": nml_aucs.mean(),
            "nml_std": nml_aucs.std(),
        }


class ResultsPlotter2:

    _protocol = "AND"
    _seed_budget = 30
    _mi_value = 0.2
    _ss_methods = [
        "deg_c",
        "deg_cd",
        "nghb_1s",
        "nghb_sd",
        "random",
    ]
    _groups = ["var_actors", "var_hubs"]
    _series = ["series_1", "series_2", "series_3", "series_4", "series_5"]

    def yield_page(self) -> Generator[tuple[str, float, int, str, str, str], None, None]:
        for page_case in product(
            [self._protocol],
            [self._mi_value],
            [self._seed_budget],
            self._ss_methods,
            self._groups,
            self._series,
        ):
            yield page_case


def load_data() -> ResultsSlicer2:
    print("loading data")
    return ResultsSlicer2(
        [
            csv_file for csv_file in glob.glob(r"data/raw_results_2nd/**/*", recursive=True)
            if re.search(r".*\.csv$", csv_file)
        ]
    )


def produce_quantitative_results(results: ResultsSlicer2) -> pd.DataFrame:
    quantitative_results_aggr = []

    for page_case in ResultsPlotter2().yield_page():
            print(page_case)

            nml_slice = results.get_slice(
                protocol=page_case[0],
                mi_value=page_case[1],
                seed_budget=page_case[2],
                ss_method=page_case[3],
                group=page_case[4],
                series=page_case[5],
            )
            mds_slice = results.get_slice(
                protocol=page_case[0],
                mi_value=page_case[1],
                seed_budget=page_case[2],
                ss_method=f"D^{page_case[3]}",
                group=page_case[4],
                series=page_case[5],
            )

            if len(nml_slice) == 0 or len(mds_slice) == 0:
                raise ValueError
            
            aucs = results.compute_auc(mds_slice=mds_slice, nml_slice=nml_slice)

            mds_gain_mean = mds_slice["gain"].mean()
            mds_gain_std = mds_slice["gain"].std()
            nml_gain_mean = nml_slice["gain"].mean()
            nml_gain_std = nml_slice["gain"].std()

            mds_auc_mean = aucs["mds_mean"]
            mds_auc_std = aucs["mds_std"]
            nml_auc_mean = aucs["nml_mean"]
            nml_auc_std = aucs["nml_std"]

            quantitative_results_aggr.append(
                {
                    "protocol": page_case[0],
                    "mi_value": page_case[1],
                    "seed_budget": page_case[2],
                    "ss_method": page_case[3],
                    "group": page_case[4],
                    "series": page_case[5],
                    "mds_gain_mean": mds_gain_mean,
                    "mds_gain_std": mds_gain_std,
                    "mds_auc_mean": mds_auc_mean,
                    "mds_auc_std": mds_auc_std,
                    "nml_gain_mean": nml_gain_mean,
                    "nml_gain_std": nml_gain_std,
                    "nml_auc_mean": nml_auc_mean,
                    "nml_auc_std": nml_auc_std,
                    "gain_winner": "mds" if mds_gain_mean > nml_gain_mean else "nml",
                    "auc_winner": "mds" if mds_auc_mean > nml_auc_mean else "nml",
                }
            )
    return pd.DataFrame(quantitative_results_aggr)


def produce_latex_table(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, index_col=0)

    df["delta_mean_gain"] = df["mds_gain_mean"] - df["nml_gain_mean"]
    df["delta_std_gain"] = np.sqrt((df["mds_gain_std"] ** 2) + (df["nml_gain_std"] ** 2))
    df["delta_mean_auc"] = df["mds_auc_mean"] - df["nml_auc_mean"]
    df["delta_std_auc"] = np.sqrt((df["mds_auc_std"] ** 2) + (df["nml_auc_std"] ** 2))
    print(df)

    df["delta_gain"] = (
        "$\scinot{" +
        df["delta_mean_gain"].apply(lambda x: f"{x:.2f}") +  
        "}{" +
        df["delta_std_gain"].apply(lambda x: f"{x:.0E}") +
        "}$"
    )
    df["delta_auc"] = (
        "$\scinot{" +
        df["delta_mean_auc"].apply(lambda x: f"{x:.2f}") +  
        "}{" +
        df["delta_std_auc"].apply(lambda x: f"{x:.0E}") +
        "}$"
    )

    df = df[["ss_method", "group", "series", "delta_gain", "delta_auc"]]

    for group in ResultsPlotter2()._groups:
        df_g = df.loc[df["group"] == group]
        for metric in ["delta_gain", "delta_auc"]:
            df_gm = df_g.pivot(index="ss_method", columns="series", values=metric)
            df_gm = df_gm.sort_index(axis=0)
            df_gm.reset_index().to_latex(output_path / f"{group}_{metric}.txt")


if __name__ == "__main__":
    # root_dir = Path(__file__).resolve().parent.parent.parent
    # out_dir = root_dir / Path("./data/processed_results_2nd/")
    # out_dir.mkdir(exist_ok=True, parents=True)
    # raw_results = load_data()
    # print(raw_results)
    # quantitative_results_df = produce_quantitative_results(raw_results)
    # quantitative_results_df.to_csv("quantitative_comparison2.csv")
    aa = Path("quantitative_comparison2.csv")
    latex_path = Path(".")
    produce_latex_table(aa, latex_path)
