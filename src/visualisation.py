import json
import re
import zipfile
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Generator

import matplotlib
import matplotlib.ticker
import network_diffusion as nd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy


class Results:

    def __init__(self, raw_results_path: str) -> None:
        self.raw_df = self.read_raw_df(raw_results_path)

    @staticmethod
    def read_raw_df(raw_result_paths: list[Path]) -> pd.DataFrame:
        dfs = []
        for csv_path in raw_result_paths:
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

    def obtain_seed_sets_for_simulated_case(
        self,
        raw_df: pd.DataFrame,
        network: str,
        protocol: str,
        seed_budget: int,
        mi_value: float,
        ss_method: str
    ) -> list[set[str]]:
        seed_sets = raw_df.loc[
            (self.raw_df["network"] == network) &
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["ss_method"] == ss_method)
        ]["seed_ids"].to_list()
        return [set(seed_set.split(";")) for seed_set in seed_sets]


class JSONParser:

    @staticmethod
    def parse_json_name(json_name):
        """Parse simulation params and match only those which utilised MDS."""
        pattern = r"^ss-(?P<ss_method>d\^.+?)--net-(?P<network>.+?)--ver-(?P<version>\d+_\d+)\.json$"
        # pattern = r"^ss-(?P<ss_method>.+?)--net-(?P<network>.+?)--ver-(?P<version>\d+_\d+)\.json$"
        match = re.match(pattern, json_name)
        if match:
            return match.groupdict()
        return {}

    def read_minimal_dominating_sets(self, zip_path):
        """Read used MDS in simulations."""
        minimal_dominating_sets = []
        with zipfile.ZipFile(zip_path, "r") as z:
            for file_name in z.namelist():
                simulation_params = self.parse_json_name(file_name)
                if simulation_params:
                    try:
                        with z.open(file_name) as f:
                            ranking_dict = json.load(f)
                            mds = [nd.MLNetworkActor.from_dict(rd).actor_id for rd in ranking_dict]
                            simulation_params["mds"] = mds
                            minimal_dominating_sets.append(simulation_params)
                    except json.JSONDecodeError:
                        print(f"Something went wrong in: {file_name}")
        return minimal_dominating_sets


def get_entropy(sets: list[set[str]]) -> float:
    """Get entropy over list of sets."""
    all_elements = [element for s in sets for element in s]
    freq = Counter(all_elements)
    probabilities = [count / sum(freq.values()) for count in freq.values()]
    return entropy(probabilities, base=2)


def analyse_set_similarity(sets: list[set[str]]) -> dict[str, float]:
    """Compute average Jaccard similarities and fratcion of unique sets."""
    if len(sets) == 0:
        return {"unique_sets_ratio": None, "jaccard_similarity": None, "entropy": None}
    unique_sets = set(frozenset(s) for s in sets)
    total_jaccard, num_comparisons = 0, 0
    for set_a, set_b in combinations(unique_sets, 2):
        jaccard = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        total_jaccard += jaccard
        num_comparisons += 1
    avg_jaccard = total_jaccard / num_comparisons if num_comparisons > 0 else None
    return {
        "unique_sets_ratio": f"{len(unique_sets)} / {len(sets)}",
        "jaccard_similarity": avg_jaccard,
        "entropy": get_entropy(sets),
    }


class Plotter:

    _protocol_and = "AND"
    _protocol_or = "OR"
    _seed_budgets_and = [15, 20, 25, 30, 35]
    _seed_budgets_or = [5, 10, 15, 20, 25]
    _mi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _ss_methods = [
        "deg_c",
        "deg_cd",
        "nghb_1s",
        "nghb_sd",
        "random",
    ]
    _networks = [   
        "aucs",
        "ckm_physicians",
        "er1",
        "er2",
        "er3",
        "er5",
        "lazega",
        "l2_course_net_1",
        "sf1",
        "sf2",
        "sf3",
        "sf5",
        "timik1q2009",
    ]

    def yield_page(self) -> Generator[tuple[str, str, str], None, None]:
        for and_case in product(
            self._networks,
            [self._protocol_and],
            self._ss_methods,
        ):
            yield and_case
        for or_case in product(
            self._networks,
            [self._protocol_or],
            self._ss_methods,
        ):
            yield or_case
    
    def yield_figure(self, protocol: str) -> Generator[tuple[int, float], None, None]:
        if protocol == "AND":
            for and_case in product(
                self._seed_budgets_and,
                self._mi_values,
            ):
                yield and_case
        elif protocol == "OR":
            for or_case in product(
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
