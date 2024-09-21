import shutil

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import network_diffusion as nd
import pandas as pd


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


def compute_gain(seeds_prct: float, coverage_prct: float) -> float:
    max_available_gain = 100 - seeds_prct
    obtained_gain = coverage_prct - seeds_prct
    return 100 * obtained_gain / max_available_gain


def save_results(result_list: list[SimulationFullResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in result_list]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


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
