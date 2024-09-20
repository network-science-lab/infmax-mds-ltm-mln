import json
import datetime
import random
import shutil

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import network_diffusion as nd
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class SimulationPartialResult:
    seed_ids: str  # IDs of actors that were seeds aggr. into string (sep. by ;)
    gain: float # gain using this seed set
    simulation_length: int  # nb. of simulation steps
    seeds: int # nb. of actors that were seeds
    exposed: int  # nb. of infected actors
    not_exposed: int  # nb. of not infected actors
    peak_infected: int  # maximal nb. of infected actors in a single sim. step
    peak_iteration: int  # a sim. step when the peak occured


@dataclass(frozen=True)
class SimulationFullResult(SimulationPartialResult):
    network: str  # network's name
    protocol: str  # protocols's name
    seed_budget: float  # a value of the maximal seed budget
    mi_value: float  # a value of the threshold
    ss_method: str  # seed selection method's name

    @classmethod
    def enhance_SPR(
        cls,
        SPR: SimulationPartialResult,
        network: str,
        protocol: str,
        seed_budget: float,
        mi_value: float,
        ss_method: str,
    ) -> "SimulationFullResult":
        return cls(
            seed_ids=SPR.seed_ids,
            gain=SPR.gain,
            simulation_length=SPR.simulation_length,
            seeds=SPR.seeds,
            exposed=SPR.exposed,
            not_exposed=SPR.not_exposed,
            peak_infected=SPR.peak_infected,
            peak_iteration=SPR.peak_iteration,
            network=network,
            protocol=protocol,
            seed_budget=seed_budget,
            mi_value=mi_value,
            ss_method=ss_method
        )


def extract_simulation_result(detailed_logs: dict[str, Any], net: nd.MultilayerNetwork) -> SimulationPartialResult:
    """Get length of diffusion, real number of seeds and final coverage."""
    simulation_length = 0
    actors_infected_total = 0
    peak_infections_nb = 0
    peak_iteration_nb = 0
    actors_nb = net.get_actors_num()

    # sort epochs indices
    epochs_sorted = sorted([int(e) for e in detailed_logs.keys()])

    # calculate metrics for each epoch
    for epoch_num in epochs_sorted:

        # obtain a number of actors in each state in the current epoch
        actorwise_log, active_actor_ids = nodewise_to_actorwise_epochlog(
            nodewise_epochlog=detailed_logs[epoch_num], actors_nb=actors_nb
        )
        activated_actors = actorwise_log["active_actors"] - actors_infected_total

        # obtain precise number and names of actors that were seeds
        if epoch_num == 0:
            seed_ids = ",".join(sorted(active_actor_ids))
            seeds = actorwise_log["active_actors"]

        # sanity check
        if epoch_num > 0:
            assert actorwise_log["active_actors"] >= actors_infected_total, \
                f"Results contradict themselves! \
                Number of active actors in {epoch_num} epoch: {actorwise_log['active_actors']} \
                number of all actors active so far: {actors_infected_total}"
    
        # update peaks
        if activated_actors > peak_infections_nb:
            peak_infections_nb = activated_actors
            peak_iteration_nb = epoch_num
        
        # update real length of diffusion
        if actorwise_log["active_actors"] != actors_infected_total:
            simulation_length = epoch_num

        # update nb of infected actors
        actors_infected_total = actorwise_log["active_actors"]

    # compute obtained gain during the simulation
    gain = compute_gain(seeds / actors_nb * 100, actors_infected_total / actors_nb * 100)

    return SimulationPartialResult(
        seed_ids=seed_ids,
        gain=gain,
        simulation_length=simulation_length,
        seeds=seeds,
        exposed=actors_infected_total,
        not_exposed=actors_nb - actors_infected_total,
        peak_infected=peak_infections_nb,
        peak_iteration=peak_iteration_nb
    )

def save_results(result_list: list[SimulationFullResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in result_list]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    for dir_path in logged_dirs:
        shutil.make_archive(logged_dirs[0].parent / dir_path.name, "zip", root_dir=str(dir_path))
    if rm_logged_dirs:
        for dir_path in logged_dirs:
            shutil.rmtree(dir_path)
    print(f"Compressed detailed logs")


def nodewise_to_actorwise_epochlog(nodewise_epochlog, actors_nb) -> tuple[dict[str, int], set[str]]:
    inactive_nodes, active_nodes = [], []
    for node_log in nodewise_epochlog:
        new_state = node_log["new_state"]
        node_name = node_log["node_name"]
        if new_state == "0":
            inactive_nodes.append(str(node_name))
        elif new_state == "1":
            active_nodes.append(str(node_name))
        else:
            raise ValueError
    actorwise_log = {
        "inactive_actors": len(set(inactive_nodes)),
        "active_actors": len(set(active_nodes)),
    }
    assert actors_nb == sum(actorwise_log.values())
    return actorwise_log, set(active_nodes)


def extract_basic_stats(detailed_logs):
    """Get length of diffusion, real number of seeds and final coverage."""
    length_of_diffusion = 0
    active_actors_num = 0
    seed_actors_num = 0
    active_nodes_list = []

    # sort epochs indices
    epochs_sorted = sorted([int(e) for e in detailed_logs.keys()])

    # calculate metrics from each epoch
    for epoch_num in epochs_sorted:

        # obtain a list and number of active nodes in current epoch
        active_nodes_epoch = []
        for node in detailed_logs[epoch_num]:
            if node["new_state"] == "1":
                active_nodes_epoch.append(node["node_name"])
        active_actors_epoch_num = len(set(active_nodes_epoch))
        
        # update real length of diffusion
        if active_actors_epoch_num != len(set(active_nodes_list)):
            length_of_diffusion = epoch_num

        # update a list of nodes that were activated during entire experiment
        active_nodes_list.extend(active_nodes_epoch)

        if epoch_num == 0:
            # obtain a pcerise number of actors that were seeds
            seed_actors_num = active_actors_epoch_num
        else:
            # sanity check to detect leaks i.e. nodes cannot be deactivated
            if active_actors_epoch_num < len(set(active_nodes_list)):
                raise AttributeError(
                    f"Results contradict themselves! \
                    Number of active actors in {epoch_num} epoch: {active_actors_epoch_num} \
                    number of all actors active so far: {len(set(active_nodes_list))}"
                )

    # get number of actors that were active at the steaady state of diffusion
    active_actors_num = len(set(active_nodes_list))

    return length_of_diffusion, active_actors_num, seed_actors_num


def compute_gain(seeds_prct, coverage_prct):
    max_available_gain = 100 - seeds_prct
    obtained_gain = coverage_prct - seeds_prct
    return 100 * obtained_gain / max_available_gain


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


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, nd.MLNetworkActor):
            return obj.__dict__
        return super().default(obj)
