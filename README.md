# Appl. of the Minimal Dominating Set for Influence Max. in Multilayer Networks

A repository with a source code for the paper: https://arxiv.org/abs/2502.15236

* Authors: Michał Czuba(¶†), Mingshan Jia(†), Piotr Bródka(¶†), Katarzyna Musial(†)
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

Dataset is stored with DVC. Thus, to obtain it you have to access a Google Drive. Please
send a request via e-mail (michal.czuba@pwr.edu.pl) to have it granted. Then, execute in shell:
`dvc pull`. **`zip` files are large and we recommend to pull them only if necessary.**

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

### Series of the results

- batch_1 real-world, g-mds, AND
- batch_2 real-world, g-mds, OR
- batch_3 artificial, g-mds, AND
- batch_4 artificial, g-mds, OR
- batch_5 timik1q2009, g-mds, AND
- batch_6 timik1q2009, g-mds, OR
- batch_7 real-world, li-mds, AND
- batch_8 real-world, li-mds, OR
- batch_9 artificial, li-mds, AND
- batch_10 artificial, li-mds, OR
- batch_11 timik1q2009, li-mds, AND
- batch_12 timik1q2009, li-mds, OR

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
    seed_budget: float      # value of the maximal seed budget
    mi_value: float         # value of the threshold
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
