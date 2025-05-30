"""A script to find the real MDS using brute-force method"""

import itertools
import uuid
from pathlib import Path

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


def main():
    nets = load_networks(NETS)
    print(nets)

    net: nd.MultilayerNetwork = nets[0].graph
    out_file = OutFile(Path(__file__).parent.parent / f"{nets[0].name}.csv")

    actors = net.get_actors()
    max_eval_size = len(actors)

    possible_ds_nb = np.sum([sp.binom(len(actors), n) for n in range(max_eval_size + 1)]).item()
    print(
        f"Number of up-to-{max_eval_size} element sets for "
        f"{len(actors)}-actor size network is {possible_ds_nb}"
    )

    p_bar = tqdm(desc="Searching possible sets", total=possible_ds_nb)
    for n in range(max_eval_size + 1):
        for cantidate_ds in itertools.combinations(actors, n):
            p_bar.update(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if is_dominating_set(set(cantidate_ds), net):
                    cds_ids = {str(a.actor_id) for a in cantidate_ds}
                    # print(f"Found {cds_ids} as a dominating set!")
                    out_file.save_ds(cds_ids)


if __name__ == "__main__":
    print("DOOPA")
    main()
