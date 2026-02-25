# robust-boed

## Computing infrastructure requirements
We have tested this codebase on Linux with Python 3.8.

## Environment setup
1. Ensure that Python and `venv` are installed.
1. Create and activate a new `venv` virtual environment as follows
```bash
python3 -m venv dad_code
source dad_code/bin/activate
```
1. Install the correct version of PyTorch following the instructions at [pytorch.org](https://pytorch.org/).
   We used `torch==1.8.0` with CUDA version 11.1.1.
1. Install the remaining package requirements using `pip install -r requirements.txt`.
1. Install the pyro at [https://github.com/pyro-ppl/pyro](https://github.com/pyro-ppl/pyro)


## Experiment: Toy Example


The method have "boed", "random","dad", and "boed_new".  and the expeiriment_id and run_id need to fill into the main.py

```bash
python3 toy_examples/dad_regression/script/fixedmodel/main.py
```

To show the degree of covariate shift and generalization error

```bash
python3 toy_examples/dad_regression/script/fixedmodel/plotting.py
```


## Experiment: Location Finding

The location_method have "boed", "random", and "boed_new".


```bash
python3 toy_examples/sources_mcmc_vi/scripts/main.py \
    --location_method "boed" \
    --grids_num 200 \
    --noise_scale 0.1 \
    --num_experiments 30 \
    --num_parallel 1 \
    --misspecification_flag False
```


To show the degree of covariate shift and generalization error

```bash
python3 toy_examples/sources_mcmc_vi/scripts/plotting.py
```
## Experiment: PK


The pk_method have "boed", "random", and new method,like "db_boed_lambda_0.5_tau_2", the new method with lambda = 0.5 and tau =2.


```bash
python3 toy_examples/pk/scripts/main.py \
    --pk_method "boed" \
    --grids_num 200 \
    --noise_scale 0.1 \
    --num_experiments 30 \
    --num_parallel 1 \
    --misspecification_flag False
```


To show the degree of covariate shift and generalization error

```bash
python3 toy_examples/pk/scripts/plotting.py
```
