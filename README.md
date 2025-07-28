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
│   ├── brute_ds            -> a brute force DS finder
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
├── visualisations_mds.py
└── mds_algos_comparison.py
```

## Doodles

Closer evaluation of Scale-Free networks

### Parameters of the `multinet` library which we were using

evolution_er_ml(n)
    n - Number of vertices (created at the beginning, before starting adding edges).

evolution_pa_ml(m0,m)
    m0 - Initial number of nodes.
    m - Number of edges created for each new vertex joining the network.

grow_ml(num_actors, num.steps, models, pr.internal, pr.external, dependency)
    num_actors - The number of actors from which new nodes are selected during the generation process.
    num_steps - Number of timestamps.
    models - A vector containing one evolutionary model for each layer to be generated (i.e., either ER or PA). Incite number of layers
    pr_internal - A vector with (for each layer) the probability that at each step the layer evolves according to the internal evolutionary model.
    pr_external - A vector with (for each layer) the probability that at each step the layer evolves importing edges from another layer.
    pr_no_action - (1 - pr_internal - pr_external), pr that in the given step nothing happens, i.e. growing of the network is slower
    dependency - A matrix LxL where element (i,j) indicates the probability that layer i will import an edge from layer j in case an external event is triggered.

### Idea

Check MDS facilitates influence maximisation in Scale-free networks with different parameters.

Select a single spreading regime to decrease a number of parameters to consider
Select the most important parameters of SF model and evaluate them. Problem -> evaluating a cartesian product of them is too demanding. Thus, I'd show such an evaluation parameter by parameter.

These parameters are fixed:
num_steps - num_actors - m0
pr_internal - 0.7 for all layers
pr_external - 0.2 for all layers
pr_no_action - 0.1 for all layers
dependency - all values eq. 1/num_layers
num_layers - 3
m0 = m

9x5

Variables:
num_actors = [500, 750, 1000, 1250, 1500]
m = [1, 3, 5, 7, 9]. # one was deleted because for 1 we have a tree
num_steps = actors - m0
