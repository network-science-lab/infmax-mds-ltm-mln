"""Main runner of the simulator."""

from pathlib import Path

import network_diffusion as nd

from src.params_handler import Network
from src.result_handler import SimulationFullResult, SimulationPartialResult
from src.runners.simulation_step import experiment_step


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
    
    # initialise necessary data regarding the network
    actors_num = net.graph.get_actors_num()
    assert len(ranking) == actors_num
    greedy_ranking: list[nd.MLNetworkActor] = []

    results = []

    # repeat until budget spent exceeds maximum value
    while (100 * len(greedy_ranking) / actors_num) <= budget[1]:

        # containers for the best actor in the run and its performance
        best_actor = None
        best_spr = SimulationPartialResult("", 0, float("inf"), 0, 0, 0, 0)

        # obtain pool of actors and limit of budget in the run
        eval_seed_budget = 100 * (len(greedy_ranking) + 1) / actors_num
        available_actors = [a for a in ranking if a not in greedy_ranking]
        # available_actors = set(ranking).difference(greedy_ranking) # faster, but nondeter. method

        for actor in available_actors:
            step_spr = experiment_step(
                protocol=proto,
                budget=(100 - eval_seed_budget, eval_seed_budget),
                mi_value=mi,
                net=net.graph,
                ranking=[
                    *greedy_ranking,
                    actor,
                    *[a for a in available_actors if a.actor_id != actor.actor_id],
                ],
                # ranking=[*greedy_ranking, actor, *available_actors.difference({actor})],  # ditto
                max_epochs_num=max_epochs_num,
                patience=patience,
                out_dir=out_dir,
            )
            if (
                (step_spr.gain > best_spr.gain) or
                (step_spr.gain == best_spr.gain and step_spr.simulation_length < best_spr.simulation_length)
            ):
                best_actor = actor
                best_spr = step_spr
        
        # when the best combination is found update rankings
        greedy_ranking.append(best_actor)
        results.append(
            SimulationFullResult.enhance_SPR(
                best_spr, net.name, proto, eval_seed_budget, mi, ss_method
            )
        )
    
    return results
