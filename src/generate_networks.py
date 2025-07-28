"""A script to generate multilayer networks."""

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
from src.aux.network_generator import MultilayerPAGenerator, convert_to_nd_and_prune, save_as_mpx
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
        for cantidate_ds in itertools.combinations(actors, n):
            p_bar.update(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if is_dominating_set(set(cantidate_ds), net_graph):
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


def generate_network_and_save(
    nb_layers: int,
    nb_actors: int,
    nb_hubs: int,
    pr_internal: float,
    pr_external: float,
    out_path: str,
) -> None:
    net_uu = MultilayerPAGenerator(
        nb_layers=nb_layers,
        nb_actors=nb_actors,
        nb_steps=nb_actors - nb_hubs,
        nb_hubs=nb_hubs,
    )(
        pr_internal=[pr_internal] * nb_layers,
        pr_external=[pr_external] * nb_layers,
    )
    net_nd = convert_to_nd_and_prune(net_uu)
    save_as_mpx(net_nd, out_path)


def run_experiments(config: dict[str, Any]) -> None:
    # get parameters of the simulation
    p_space = params_handler.get_parameter_space_generator(
        nb_layers=config["parameters"]["nb_layers"],
        nb_actors=config["parameters"]["nb_actors"],
        nb_hubs=config["parameters"]["nb_hubs"],
        pr_internal=config["parameters"]["pr_internal"],
        pr_external=config["parameters"]["pr_external"],
    )
    repetitions = config["run"]["repetitions"]

    # prepare output directories and save the config
    out_dir = params_handler.create_out_dir(config["logging"]["out_dir"])
    cohort_name: str = config["logging"]["cohort_name"]
    config["git_sha"] = utils.get_recent_git_sha()
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    # generate the networks
    for p_idx, p_case in enumerate(p_space, 1):
        p_str = "nb_layers:{0}-nb_actors:{1}-nb_hubs:{2}-pr_internal:{3}-pr_external:{4}".format(
            p_case[0], p_case[1], p_case[2], p_case[3], p_case[4],
        )
        print(p_str)
        p_dir = out_dir / f"{cohort_name}_{p_idx}"
        p_dir.mkdir(exist_ok=True, parents=True)
        with open(p_dir / "info.txt", "w", encoding="utf-8") as f:
            f.write(p_str)        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    generate_network_and_save,
                    p_case[0],
                    p_case[1],
                    p_case[2],
                    p_case[3],
                    p_case[4],
                    f"{str(p_dir)}/{r}.mpx"
                ) for r in range(1, repetitions + 1)
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
