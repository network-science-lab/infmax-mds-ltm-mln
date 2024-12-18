"""A script with defined single simulation step."""

from pathlib import Path

import network_diffusion as nd

from src.result_handler import SimulationPartialResult, extract_simulation_result
from src.models.mltm import MDSLimitedMLTModel

def experiment_step(
    protocol: str,
    budget: int,
    mi_value: float,
    net: nd.MultilayerNetwork,
    ranking: list[nd.MLNetworkActor],
    max_epochs_num: int, 
    patience: int,
    out_dir: Path | None,
) -> SimulationPartialResult:

    # initialise spreading model
    mltm = MDSLimitedMLTModel(
        protocol=protocol,
        seed_selector=nd.seeding.MockingActorSelector(ranking),
        seeding_budget = budget,
        mi_value=mi_value,
    )

    # run experiment on a deep copy of the network!
    experiment = nd.Simulator(model=mltm, network=net.copy())
    logs = experiment.perform_propagation(n_epochs=max_epochs_num, patience=patience)

    # extract global results
    simulation_result = extract_simulation_result(logs.get_detailed_logs(), net)

    if out_dir:
        logs.report(path=str(out_dir))
    return simulation_result
