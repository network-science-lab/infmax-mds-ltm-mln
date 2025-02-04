# Inf. Max. with Minimal Dominating Set under LTM for Multilayer Networks

A repository to check efficiency of MDS-based seed selection methods in influence maximisation
problem under Multilayer Linear Threshold Model.

* Authors: Michał Czuba(¶†), Mingshan Jia(†), Kaska Gabrys-Musial(†), Piotr Bródka(¶†)
* Affiliation:  
        (¶) WUST, Wrocław, Lower Silesia, Poland  
        (†) UTS, Sydney, NSW, Australia

## Configuration of the runtime

First, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-mds-ltm-mln
python -m ipykernel install --user --name=infmax-mds-ltm-mln
```

To use scripts which produce analysis, install the source code:

```bash
pip install -e .
```

## Data

Dataset is stored on a DVC remote. Thus, to obtain it you have to access a Google Drive. Please
send a request via e-mail (michal.czuba@pwr.edu.pl) to have it granted. Then, simply execute in
the shell: `dvc pull`. **The dataset is large, hence we recommend to pull `zip` files only if
necessary.** For normal usage it is engouh to pull networks (`dvc pull data/networks`) and raw
results which are subjects of the analysis (that can be done in two ways - either pull all results
and kill the disk: `dvc pull data/raw_results` or just pre-preprocessed data with configs:
`sh data/get_raw_results_slim.sh`).

To extract raw results and pack it into separate `zip` file run: `sh data/zip_raw_results_slim.sh`

## Structure of the repository

```bash
.
├── README.md
├── data
│   ├── networks            -> networks used in exmeriments
│   ├── processed_results
│   ├── raw_results
│   └── test                -> examplary results of the simulator used in the E2E test
├── env                     -> a definition of the runtime environment
├── scripts
│   ├── analysis
│   └── configs             -> exemplary configuration files
├── src                     -> scripts to execute experiments and process the results
├── run_experiments.py      -> an entrypoint to trigger the pipeline to evaluate MDS in InfMax
├── test_reproducibility.py -> E2E test to prove that results can be repeated
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

## Obtaining analysis of results

To process raw results please execute scripts in `scripts/analysis` directory in the order as 
depicted in a following tree. Please note, that names of scripts reflect names of genreated files
under `data/processed_results`:

```bash
.
├── distr_expos.ipynb
├── quantitative_comparison.py
│   ├── effectiveness_heatmaps.py
│   └── profile_reports.py
├── metrics.py
├── similarities_mds.py
├── similarities_seeds.py
└── visualisations_mds.py
```
