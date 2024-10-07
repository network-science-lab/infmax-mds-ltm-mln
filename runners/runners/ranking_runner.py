"""Pure ranking based step handler."""

from pathlib import Path

import network_diffusion as nd

from runners.params_handler import Network
from runners.result_handler import SimulationFullResult
from runners.simulation_step import experiment_step


def handle_step(
    proto: str, 
    budget: tuple[float, float],
    mi: float,
    net: Network,
    ss_method: str,
    ranking: list[nd.MLNetworkActor],
    max_epochs_num: int,
    patience: int,
    out_dir: Path | None,
) -> list[SimulationFullResult]:
    """The easiest way to handle case basing only on the ranking."""
    step_spr = experiment_step(
        protocol=proto,
        budget=budget,
        mi_value=mi,
        net=net.graph,
        ranking=ranking,
        max_epochs_num=max_epochs_num,
        patience=patience,
        out_dir=out_dir,
    )
    step_sfr = SimulationFullResult.enhance_SPR(step_spr, net.name, proto, budget[1], mi, ss_method)
    return [step_sfr]
