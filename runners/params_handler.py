"""A script with functions to facilitate liading simulation's parameters and input data."""

import itertools
import json

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

import network_diffusion as nd

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from runners.new_selectors import DCBSelector


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


def get_parameter_space(
    protocols: list[str],
    seed_budgets: list[float],
    mi_values: list[str],
    networks: list[str],
    ss_methods: list[str],
) -> list[tuple[str, tuple[int, int], float, str, SeedSelector]]:
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    return list(itertools.product(protocols, seed_budgets_full, mi_values, networks, ss_methods))


def get_with_mds(get_ss_func: Callable) -> Callable:
    """Decorate seed selection loader function so that it can optionally use MDS."""
    @wraps(get_ss_func)
    def wrapper(selector_name: str) -> nd.seeding.BaseSeedSelector:
        if selector_name[:2] == "d^":
            ss_method = get_ss_func(selector_name[2:])
            return nd.seeding.DriverActorSelector(method=ss_method)
        return get_ss_func(selector_name)
    return wrapper


@get_with_mds
def get_seed_selector(selector_name: str) -> nd.seeding.BaseSeedSelector:
    if selector_name == "btw":
        return nd.seeding.BetweennessSelector()
    if selector_name == "cbim":
        return nd.seeding.CBIMSeedselector(merging_idx_threshold=1)
    elif selector_name == "cim":
        return nd.seeding.CIMSeedSelector()
    elif selector_name == "cls":
        return nd.seeding.ClosenessSelector()
    elif selector_name == "dcb":
        return DCBSelector()
    elif selector_name == "deg_c":
        return nd.seeding.DegreeCentralitySelector()
    elif selector_name == "deg_cd":
        return nd.seeding.DegreeCentralityDiscountSelector()
    elif selector_name == "k_sh":
        return nd.seeding.KShellSeedSelector()
    elif selector_name == "k_sh_m":
        return nd.seeding.KShellMLNSeedSelector()
    elif selector_name == "kpp_sh":
        return nd.seeding.KPPShellSeedSelector()
    elif selector_name == "nghb_1s":
        return nd.seeding.NeighbourhoodSizeSelector(connection_hop=1)
    elif selector_name == "nghb_2s":
        return nd.seeding.NeighbourhoodSizeSelector(connection_hop=2)
    elif selector_name == "nghb_sd":
        return nd.seeding.NeighbourhoodSizeDiscountSelector()
    elif selector_name == "p_rnk":
        return nd.seeding.PageRankSeedSelector()
    elif selector_name == "p_rnk_m":
        return nd.seeding.PageRankMLNSeedSelector()
    elif selector_name == "random":
        return nd.seeding.RandomSeedSelector()
    elif selector_name == "v_rnk":
        return nd.seeding.VoteRankSeedSelector()
    elif selector_name == "v_rnk_m":
        return nd.seeding.VoteRankMLNSeedSelector()
    raise AttributeError(f"{selector_name} is not a valid seed selector name!")


def load_networks(networks: list[str]) -> list[Network]:
    nets = []
    for net_name in networks:
        print(f"Loading {net_name} network")
        nets.append(Network(net_name, load_network(net_name=net_name, as_tensor=False)))
    return nets


def load_seed_selectors(ss_methods: list[str]) -> list[SeedSelector]:
    ssms = []
    for ssm_name in ss_methods:
        print(f"Initialising seed selection method: {ssm_name}")
        ssms.append(SeedSelector(ssm_name, get_seed_selector(ssm_name)))
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
            ss_ranking_name = Path(f"ss-{ssm.name}--net-{net.name}--ver-{version}.json")

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
