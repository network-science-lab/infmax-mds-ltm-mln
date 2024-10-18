# Inf. Max. with Minimal Dominating Set under LTM for Multilayer Networks

A repository to check efficiency of MDS-based seed selection methods in influence maximisation
problem under Multilayer Linear Threshold Model.

* Authors: Piotr Bródka*, Michał Czuba*, Mingshan Jia**
* Affiliation: *WUST, Wrocław, Lower Silesia, Poland; **UTS, Sydney, NSW, Australia

## Configuration of the runtime

First, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-mds-ltm-mln
```

## Data

Dataset is stored on a DVC remote. Thus, to obtain it you have to access a Google Drive. Please
send a request via e-mail (michal.czuba@pwr.edu.pl) to have it granted. Then, simply execute in
the shell: `dvc pull`

## Structure of the repository
```
.
├── data                    -> networks to compute actors' marginal efficiency for + results
├── env                     -> a definition of the runtime environment
├── src                     -> scripts to execute experiments according to provided configs
├── example_config.yaml     -> an example of the config accepted by the simulator
├── README.md
├── run_experiments.py      -> main entrypoint to trigger the pipeline
└── test_reproducibility.py -> E2E test to prove that results can be repeated
```

## Running the pipeline

To run experiments execute: `python run_experiments.py <config file>`. See `example_config.yaml` for
inspirations. As a result, for each repetition of the cartesian product computed for the provided
parameters, a csv file will be obtained with following columns:

```python
{
    seed_ids: str           # IDs of actors that were seeds aggr. into string (sep. by ;)
    gain: float             # gain* obtained using this seed set
    simulation_length: int  # nb. of simulation steps
    seed_nb: int            # nb. of actors that were seeds
    exposed_nb: int         # nb. of active actors at the end of the simulation
    unexposed_nb: int       # nb. of actors that remained inactive
    expositons_rec: str     # record of new activations in each epoch aggr. into string (sep. by ;)
    network: str            # network's name
    protocol: str           # protocols's name
    seed_budget: float      # a value of the maximal seed budget
    mi_value: float         # a value of the threshold
    ss_method: str          # seed selection method's name
}
```

`*` Gain is the percentage of the non-initially seeded population that became exposed during the
simulation: `(exposed_nb - seed_nb) / (total_actor_nb - seed_nb) * 100%`

The simulator will also save provided configuraiton file, rankings of actors used in computations,
and detailed logs of evaluated cases whose index divided modulo by `full_output_frequency` equals 0.


## Results reproducibility

Results are supposed to be fully reproducable. There is a test for that: `test_reproducibility.py`.
