"""Main runner of the simulator."""

import yaml
from pathlib import Path
from typing import Any

from runners.utils import (
    SimulationFullResult,
    get_current_time,
    get_diff_of_times,
    save_results,
    zip_detailed_logs,
)
from tqdm import tqdm
from runners import commons, step_classic

DET_LOGS_DIR = "detailed_logs"
RANKINGS_DIR = "rankings"


def run_experiments(config: dict[str, Any]) -> None:

    # load networks, compute rankings and save them
    nets = commons.load_networks(config["networks"])
    ssms = commons.load_seed_selectors(config["model"]["parameters"]["ss_methods"])

    # get parameters of the simulation
    p_space = commons.get_parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        seed_budgets=config["model"]["parameters"]["seed_budgets"],
        mi_values=config["model"]["parameters"]["mi_values"],
        networks=[n.name for n in nets],
        ss_methods=[s.name for s in ssms],
    )

    # get parameters of the simulator
    logging_freq = config["logging"]["full_output_frequency"]
    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    ranking_path = config.get("ranking_path")
    repetitions = config["run"]["repetitions"]
    rng_seed = config["run"]["random_seed"]

    # prepare output directories and determine how to store results
    out_dir = Path(config["logging"]["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    det_dir = out_dir / DET_LOGS_DIR
    det_dir.mkdir(exist_ok=True, parents=True)
    rnk_dir = out_dir / RANKINGS_DIR
    rnk_dir.mkdir(exist_ok=True, parents=True)
    compress_to_zip = config["logging"]["compress_to_zip"]

    # save the config
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # get a start time
    start_time = get_current_time()
    print(f"Experiments started at {start_time}")

    # repeat main loop for given number of times
    for rep in range(1, repetitions + 1):
        print(f"\n\nRepetition {rep}/{repetitions}\n\n")
        rep_results = []

        # for each network ans ss method compute a ranking
        rankings = commons.compute_rankings(
            seed_selectors=ssms,
            networks=nets,
            out_dir=rnk_dir,
            version=f"{rng_seed}_{rep}",
            ranking_path=ranking_path,
        )

        # start simulations
        p_bar = tqdm(p_space, desc="", leave=False, colour="green")
        for idx, investigated_case in enumerate(p_bar):
            proto, budget, mi, net_name, ss_method = investigated_case
            p_bar.set_description_str(
                commons.get_case_name_rich(
                    rep_idx=rep,
                    reps_nb=repetitions,
                    case_idx=idx,
                    cases_nb=len(p_bar),
                    protocol=proto,
                    mi_value=mi,
                    budget=budget[1],
                    net_name=net_name,
                    ss_name=ss_method,
                )
            )
            ic_name = f"{commons.get_case_name_base(proto, mi, budget[1], ss_method, net_name)}_{rep}"
            try:
                step_spr = step_classic.experiment_step(
                    protocol=proto,
                    budget=budget,
                    mi_value=mi,
                    net=[net.graph for net in nets if net.name == net_name][0],
                    ranking=rankings[(net_name, ss_method)],
                    max_epochs_num=max_epochs_num,
                    patience=patience,
                    out_dir=det_dir / ic_name if rep % logging_freq == 0 else None,
                )
                step_sfr = SimulationFullResult.enhance_SPR(step_spr, net_name, proto, budget[1], mi, ss_method)
                rep_results.append(step_sfr)
            except BaseException as e:
                print(f"\nExperiment failed for case: {ic_name}")
                raise e
        
        # aggregate results for given repetition number and save them to a csv file
        save_results(rep_results, out_dir / f"results_{rep}.csv")

    # compress global logs and config
    if compress_to_zip:
        zip_detailed_logs([det_dir, rnk_dir], rm_logged_dirs=True)

    finish_time = get_current_time()
    print(f"Experiments finished at {finish_time}")
    print(f"Experiments lasted {get_diff_of_times(start_time, finish_time)} minutes")
