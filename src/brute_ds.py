"""A script to find the real MDS using brute-force method"""

import concurrent.futures
import itertools
import uuid
import warnings
from pathlib import Path
from typing import Any

import network_diffusion as nd
import numpy as np
import scipy.special as sp
import yaml
from tqdm import tqdm

from src import params_handler, utils
from src.models.mds.utils import is_dominating_set


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


def get_possible_ds_nb(nb_actors: int, min_eval_size: int,  max_eval_size: int) -> int:
    """Get number of solutions to evaluate."""
    possible_ds_nb = np.sum(
            [sp.binom(nb_actors, n) for n in range(min_eval_size, max_eval_size + 1)]
        ).astype(int).item()
    print(
        f"Number of {min_eval_size}-to-{max_eval_size} elemement sets "
        f"for {nb_actors} size net is {possible_ds_nb}."
    )
    return possible_ds_nb


def find_real_mds(
    net_graph: nd.MultilayerNetwork,
    net_name: str,
    min_eval_size: int,
    max_eval_size: int,
    out_dir: Path,
) -> None:
    """Search all possible combinations of actors up to `max_eval_size` to find dominating sets."""
    actors = net_graph.get_actors()
    possible_ds_nb = get_possible_ds_nb(len(actors), min_eval_size, max_eval_size)
    out_file = OutFile(out_dir / f"{net_name}.csv")
    p_bar = tqdm(desc=f"Searching possible sets {net_name}", total=possible_ds_nb)
    for n in range(min_eval_size, max_eval_size + 1):
        print(n)
        for cantidate_ds in itertools.combinations(actors, n):
            p_bar.update(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if is_dominating_set(set(cantidate_ds), net_graph):
                    print(f"Found MDS at {n} size!")
                    cds_ids = {str(a.actor_id) for a in cantidate_ds}
                    out_file.save_ds(cds_ids)


def _process_net(net: params_handler.Network, out_dir: Path, min_es: int, max_es: int) -> None:
    print(f"Processing {net.name} network")
    find_real_mds(
        net_graph=net.graph,
        net_name=net.name,
        min_eval_size=min_es,
        max_eval_size=max_es,
        out_dir=out_dir,
    )
    print(f"Computations completed for: {net.name}")


def run_experiments(config: dict[str, Any]) -> None:
    networks = params_handler.load_networks(config["networks"])
    out_dir = params_handler.create_out_dir(config["logging"]["out_dir"])
    min_es, max_es = config["es_size_range"]
    config["git_sha"] = utils.get_recent_git_sha()
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_net, net, out_dir, min_es, max_es) for net in networks]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"An error occurred: {e}")
