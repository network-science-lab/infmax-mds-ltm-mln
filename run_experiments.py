import argparse
import yaml

from src import main, brute_ds, generate_networks
from src.utils import set_rng_seed


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        nargs="?",
        type=str,
        # default="scripts/configs/example_main.yaml",
        # default="scripts/configs/example_bruteds.yaml",
        default="scripts/configs/example_generate.yaml",
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")

    if random_seed := config["run"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_rng_seed(config["run"]["random_seed"])

    if (experiment_type := config["run"].get("experiment_type")) == "main":
        entrypoint = main
    elif experiment_type == "brute_ds":
        entrypoint = brute_ds
    elif experiment_type == "generate_networks":
        entrypoint = generate_networks
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")

    entrypoint.run_experiments(config)
