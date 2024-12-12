import itertools
from pathlib import Path
from typing import Generator
from matplotlib import pyplot as plt

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd



class Results:

    def __init__(self, raw_results_path: str) -> None:
        self.raw_df = self.read_raw_df(raw_results_path)

    @staticmethod
    def read_raw_df(raw_results_path: str) -> pd.DataFrame:
        dfs = []
        for csv_path in list(Path(raw_results_path).glob("**/*.csv")):
            csv_df = pd.read_csv(csv_path)
            dfs.append(csv_df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_slice(
        self, protocol: str, mi_value: float, seed_budget: int, network:str, ss_method: str
    ) -> pd.DataFrame:
        slice_df = self.raw_df.loc[
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["network"] == network) &
            (self.raw_df["ss_method"] == ss_method)
        ].copy()
        slice_df["expositions_rec"] = slice_df["expositions_rec"].map(lambda x: [int(xx) for xx in x.split(";")])
        return slice_df.reindex()

    @staticmethod
    def mean_expositions_rec(slice_df: pd.DataFrame) -> dict[str, np.array]:
        max_len = max(slice_df["expositions_rec"].map(lambda x: len(x)))
        exp_recs_padded = np.zeros([len(slice_df), max_len])
        for run_idx, step_idx in enumerate(slice_df["expositions_rec"]):
            exp_recs_padded[run_idx][0:len(step_idx)] = step_idx
        return {
            "avg": np.mean(exp_recs_padded, axis=0).round(3),
            "std": np.std(exp_recs_padded, axis=0).round(3),
            "cdf": np.cumsum(np.mean(exp_recs_padded, axis=0)).round(3),
        }

    @staticmethod
    def get_actors_nb(slice_df: np.ndarray) -> np.ndarray:
        return (slice_df.iloc[0]["exposed_nb"] + slice_df.iloc[0]["unexposed_nb"]).astype(int).item()



class Plotter:

    _protocol_and = "AND"
    _protocol_or = "OR"
    _seed_budgets_and = [15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    _seed_budgets_or = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    _mi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _ss_methods = [
        "deg_c",
        # "deg_cd",
        "nghb_sd",
        # "p_rnk",
        # "p_rnk_m",
        "random",
        # "v_rnk",
        # "v_rnk_m",
    ]
    _networks = [   
        # "arxiv_netscience_coauthorship",
        "aucs",
        "ckm_physicians",
        # "er2",
        # "er3",
        # "er5",
        "eu_transportation",
        # "eu_transport_klm",
        "lazega",
        "l2_course_net_1",
        # "l2_course_net_2",
        # "l2_course_net_3",
        # "sf2",
        # "sf3",
        # "sf5",
        # "toy_network",
        # "timik1q2009",
    ]

    def yield_page(self) -> Generator[tuple[str, str, str], None, None]:
        for and_case in itertools.product(
            self._networks,
            [self._protocol_and],
            self._ss_methods,
        ):
            yield and_case
        for or_case in itertools.product(
            self._networks,
            [self._protocol_or],
            self._ss_methods,
        ):
            yield or_case
    
    def yield_figure(self, protocol: str) -> Generator[tuple[int, float], None, None]:
        if protocol == "AND":
            for and_case in itertools.product(
                self._seed_budgets_and,
                self._mi_values,
            ):
                yield and_case
        elif protocol == "OR":
            for or_case in itertools.product(
                self._seed_budgets_or,
                self._mi_values,
            ):
                yield or_case
        else:
            raise AttributeError(f"Unknown protocol {protocol}!")

    @staticmethod
    def plot_avg_with_std(record: list[dict[str, float]], ax: matplotlib.axes.Axes, label: str, colour: str):
        y_avg = record["cdf"]
        y_std = record["std"]
        x = np.arange(len(y_avg))
        ax.plot(x, y_avg, label=label, color=colour)
        ax.fill_between(x, y_avg - y_std, y_avg + y_std, color=colour, alpha=0.4)
    

    def plot_single_comparison(
        self,
        record_mds: list[dict[str, float]],
        record_nml: list[dict[str, float]],
        actors_nb: int,
        mi_value: float,
        seed_budget: int,
        ax: matplotlib.axes.Axes,
    ) -> None:
        plt.rc("legend", fontsize=8)
        x_max = max(len(record_mds["avg"]), len(record_nml["avg"])) - 1
        self.plot_avg_with_std(record_mds, ax, "MDS", "greenyellow")
        self.plot_avg_with_std(record_nml, ax, "NML", "sandybrown")
        ax.hlines(y=actors_nb, xmin=0, xmax=x_max, color="red")
        # ax.set_xlabel("Step")
        # ax.set_ylabel("Expositions")
        ax.set_xlim(left=0, right=x_max)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_visible(False)
        ax.legend(loc="lower right")
        ax.set_title(f"mu={mi_value}, |S|={seed_budget}")

    @staticmethod
    def plot_dummy_fig(mi_value: float, seed_budget: int, ax: matplotlib.axes.Axes) -> None:
        ax.set_title(f"No results for mu={mi_value}, |S|={seed_budget}")
