"""A script to find the real MDS using brute-force method"""

import itertools
import uuid
from pathlib import Path
from typing import Any

import warnings

import network_diffusion as nd
import numpy as np
import scipy.special as sp
from tqdm import tqdm

from src.models.mds.utils import is_dominating_set
from src.params_handler import load_networks

NETS = ["toy_network"] # , "aucs"]


class OutFile:
    """Class to handle found greedily dominating sets."""

    def __init__(self, out_path: Path) -> None:
        if not out_path.parent.exists():
            out_path.parent.mkdir(exist_ok=True, parents=True)
        while out_path.exists():
            warnings.warn("Desired out_file exists - creating a new one")
            old_name = out_path.stem
            candidate_name = old_name + f"_{uuid.uuid4().hex[:8]}" + out_path.suffix
            out_path = out_path.parent / candidate_name
        print(f"Saving results in {out_path}")
        self.out_path = out_path
        self._init_header()
        
    def _init_header(self) -> None:
        with open(self.out_path, mode="w") as file:
            file.write("dominating_set,length\n")

    def save_ds(self, ds: set[str]) -> None:
        with open(self.out_path, mode="a") as file:
            file.write(f"{';'.join(ds)},{len(ds)}\n")


def get_possible_ds_nb(nb_actors: int, max_eval_size: int) -> int:
    """Get number of solutions to evaluate."""
    possible_ds_nb = int(np.sum([sp.binom(nb_actors, n) for n in range(max_eval_size + 1)]).item())
    print(f"Nb of up-to-{max_eval_size} elem. sets for {nb_actors} size net is {possible_ds_nb}")
    return possible_ds_nb


def find_real_mds(
    net_graph: nd.MultilayerNetwork,
    net_name: str,
    max_eval_size: int,
    out_dir: Path,
) -> None:
    """Search all possible combinations of actors up to `max_eval_size` to find dominating sets."""
    actors = net_graph.get_actors()
    possible_ds_nb = get_possible_ds_nb(len(actors), max_eval_size)
    out_file = OutFile(out_dir / f"{net_name}.csv")
    p_bar = tqdm(desc="Searching possible sets", total=possible_ds_nb)
    for n in range(max_eval_size + 1):
        for cantidate_ds in itertools.combinations(actors, n):
            p_bar.update(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if is_dominating_set(set(cantidate_ds), net_graph):
                    cds_ids = {str(a.actor_id) for a in cantidate_ds}
                    out_file.save_ds(cds_ids)


def main():
    nets = load_networks(NETS)
    for net in nets:
        print(f"Processing {net.name} network")
        find_real_mds(
            net_graph=net.graph,
            net_name=net.name,
            max_eval_size=net.graph.get_actors_num(),
            out_dir=Path(__file__).parent.parent,
        )


def run_experiments(config: dict[str, Any]) -> None:
    print("hei-di hei-do hei-da!")



if __name__ == "__main__":
    print("DOOPA")
    main()
