"""A script with functions to facilitate liading simulation's parameters and input data."""

import itertools
import json
import math
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

import network_diffusion as nd

from src.loaders.net_loader import load_network
from src.models.seed_selectors import DCBSelector, DriverActorLimitedSelector


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
) -> tuple[list[tuple[str, tuple[int, int], float, str, str]], str]:
    runner_type = determine_runner(ss_methods)
    print(f"Determined runner type: {runner_type}")
    if runner_type == "greedy":
        seed_budgets = [max(seed_budgets)]
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    p_space = itertools.product(protocols, seed_budgets_full, mi_values, networks, ss_methods)
    return list(p_space), runner_type


def determine_runner(ss_methods: list[str]):
    ssm_prefixes = [ssm[:2] == "g^" for ssm in ss_methods]
    if all(ssm_prefixes):
        return "greedy"
    elif not any(ssm_prefixes):
        return "ranking"
    raise ValueError(f"Config file shall contain ssm that can be run with one runner {ss_methods}!")


def get_logging_frequency(full_output_frequency: int) -> float | int:
    if full_output_frequency == -1:
        return math.pi
    return full_output_frequency


def create_out_dir(out_dir: str) -> Path:
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        print("Redirecting output to hell...")
        out_dir = Path(tempfile.mkdtemp())
    return out_dir


def get_for_greedy(get_ss_func: Callable) -> Callable:
    """Decorate seed selection loader so that it can determine a base ranking for greedy."""
    @wraps(get_ss_func)
    def wrapper(selector_name: str) -> nd.seeding.BaseSeedSelector:
        if selector_name[:2] == "g^":
            return get_ss_func(selector_name[2:])
        return get_ss_func(selector_name)
    return wrapper


def get_with_mds(get_ss_func: Callable) -> Callable:
    """Decorate seed selection loader so that it can optionally use MDS."""
    @wraps(get_ss_func)
    def wrapper(selector_name: str) -> nd.seeding.BaseSeedSelector:
        if selector_name[:2] == "d^":
            ss_method = get_ss_func(selector_name[2:])
            return DriverActorLimitedSelector(method=ss_method, return_only_mds=True)
        return get_ss_func(selector_name)
    return wrapper


@get_for_greedy
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


# TODO: this function is not able yet to treat rankings for method: g^random and random as the the
# same one. We can try to implement such functionality to speed up computations
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
