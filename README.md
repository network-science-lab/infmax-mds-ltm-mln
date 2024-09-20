# Inf. Max. with Minimal Dominating Set under LTM for Multilayer Networks

A repository to check efficiency of MDS-based seed selection methods in influence maximisation
problem.

* Authors: Piotr Bródka, Michał Czuba
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

First, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-mds-ltm-mln

```

Then, pull the submodule and install its code:

```bash
git submodule init && git submodule update
pip install -e _data_set
```

## Data

Dataset is stored in a separate repository bounded with this project as a git submodule. Thus, to
obtain it you have to pull the data from the DVC remote. In order to access it, please sent a
request to get  an access via  e-mail (michal.czuba@pwr.edu.pl). Then, simply execute in a shell:
* `cd _data_set && dvc pull nsl_data_sources/raw/multi_layer_networks/*.dvc && cd ..`

## Structure of the repository
```
.
├── _configs                -> def. of the spreading regimes under which do computations
├── _data_set               -> networks to compute actors' marginal efficiency for
├── _test_data              -> examplary outputs of the dataset generator used in the E2E test
├── _output                 -> a directory where we recommend to save results
├── env                     -> a definition of the runtime environment
├── runners                 -> scripts to execute experiments according to provided configs
├── README.md          
├── run_experiments.py      -> main entrypoint to trigger the pipeline
└── test_reproducibility.py -> E2E test to prove that results can be repeated
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to 
the configuration file. See examples in `_config/examples` for inspirations. As a result, for each
evaluated spreading case, a csv file will be obtained with TODO

## Results reproducibility

Results are supposed to be fully reproducable. There is a test for that: `test_reproducibility.py`.
