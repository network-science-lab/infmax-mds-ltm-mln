import itertools
import json
import random
import warnings

from pathlib import Path
from typing import Any

import torch
import network_diffusion as nd
import numpy as np

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from dataclasses import dataclass
from math import log10


warnings.filterwarnings(action="ignore", category=FutureWarning)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, nd.MLNetworkActor):
            return obj.__dict__
        return super().default(obj)


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


@dataclass(frozen=True)
class SeedSelector:
    name: str
    selector: nd.seeding.BaseSeedSelector


def set_rng_seed(seed: int) -> None:
    """Fix seed of the random numbers generator for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameter_space(
    protocols: list[str],
    seed_budgets: list[float],
    mi_values: list[str],
    networks: list[str],
    ss_methods: list[str],
) -> list[tuple[str, tuple[int, int], float, str, SeedSelector]]:
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    return list(itertools.product(protocols, seed_budgets_full, mi_values, networks, ss_methods))


def get_case_name_base(protocol: str, mi_value: float, budget: int, ss_name: str, net_name: str) -> str:
    return f"proto-{protocol}--mi-{round(mi_value, 3)}--budget-{budget}--ss-{ss_name}--net-{net_name}"


def get_case_name_rich(
    case_idx: int,
    cases_nb: int,
    rep_idx: int,
    reps_nb: int,
    protocol: str,
    mi_value: float,
    budget: int, 
    net_name: str,
    ss_name: str,
) -> str:
    return (
        f"repet-{str(rep_idx).zfill(int(log10(reps_nb)+1))}/{reps_nb}--" +
        f"case-{str(case_idx).zfill(int(log10(cases_nb)+1))}/{cases_nb}--" +
        get_case_name_base(protocol, mi_value, budget, ss_name, net_name)
    )


def get_seed_selector(selector_name):
    if selector_name == "cbim":
        return nd.seeding.CBIMSeedselector
    elif selector_name == "cim":
        return nd.seeding.CIMSeedSelector
    elif selector_name == "degree_centrality":
        return nd.seeding.DegreeCentralitySelector
    elif selector_name == "degree_centrality_discount":
        return nd.seeding.DegreeCentralityDiscountSelector
    elif selector_name == "k_shell":
        return nd.seeding.KShellSeedSelector
    elif selector_name == "k_shell_mln":
        return nd.seeding.KShellMLNSeedSelector
    elif selector_name == "kpp_shell":
        return nd.seeding.KPPShellSeedSelector
    elif selector_name == "neighbourhood_size":
        return nd.seeding.NeighbourhoodSizeSelector
    elif selector_name == "neighbourhood_size_discount":
        return nd.seeding.NeighbourhoodSizeDiscountSelector
    elif selector_name == "page_rank":
        return nd.seeding.PageRankSeedSelector
    elif selector_name == "page_rank_mln":
        return nd.seeding.PageRankMLNSeedSelector
    elif selector_name == "random":
        return nd.seeding.RandomSeedSelector
    elif selector_name == "vote_rank":
        return nd.seeding.VoteRankSeedSelector
    elif selector_name == "vote_rank_mln":
        return nd.seeding.VoteRankMLNSeedSelector
    raise AttributeError(f"{selector_name} is not a valid seed selector name!")


def load_networks(networks: list[str]) -> list[Network]:
    nets = []
    for net_name in networks:
        print(f"Loading {net_name} network")
        nets.append(Network(net_name, load_network(net_name=net_name, as_tensor=False)))
    return nets


def load_seed_selectors(ss_methods: list[dict[str, Any]]) -> list[SeedSelector]:
    ssms = []
    for ss_method in ss_methods:
        print(f"Initialising seed selection method: {ss_method['name']}")
        ss_class = get_seed_selector(ss_method["name"])
        ss_params = ss_method['parameters']
        ss_name = f"{ss_class.__name__}::" + "".join([f"{k}:{v}" for k, v in ss_params.items()])
        ssms.append(SeedSelector(ss_name, ss_class(**ss_params)))
    return ssms


def compute_rankings(
    seed_selectors: list[SeedSelector],
    networks: list[Network],
    out_dir: Path,
    version: int,
    ranking_path: Path | None = None,
) -> dict[tuple[str, str]: list[nd.MLNetworkActor]]:
    """For given networks and seed seleciton methods compute or load rankings of actors."""
    
    nets_and_ranks = {}  # {(net_name, ss_name): ranking}
    for n_idx, net in enumerate(networks):
        print(f"Computing ranking for: {net.name} ({n_idx+1}/{len(networks)})")

        for s_idx, ssm in enumerate(seed_selectors):
            print(f"Using method: {ssm.name} ({s_idx+1}/{len(seed_selectors)})")   
            ss_ranking_name = Path(f"{net.name}-{ssm.name}-{version}.json")

            # obtain ranking for given ssm and net
            if ranking_path:
                ranking_file = Path(ranking_path) / ss_ranking_name
                with open(ranking_file, "r") as f:
                    ranking_dict = json.load(f)
                ranking = [nd.MLNetworkActor.from_dict(rd) for rd in ranking_dict]
                print("\tranking loaded")
            else:
                ranking = ssm.selector(net.graph, actorwise=True)
                print("\tranking computed")
            nets_and_ranks[(net.name, ssm.name)] = ranking

            # save computed ranking
            with open(out_dir / ss_ranking_name, "w") as f:
                json.dump(ranking, f, cls=JSONEncoder)
                print(f"\tranking saved in the storage")

    return nets_and_ranks
