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

## MLFlow
We use `mlflow` to log metric and store network parameters. Each experiment run is stored in
a directory `mlruns` which will be created automatically. Each experiment is assigned a
numerical `<ID>` and each run gets a unique `<HASH>`.

## Experiment: Toy Example
To train a DAD network for toy example under well-specified model in the paper,

```bash
python3 toy_examples/dad_regression/script/train_dad_regression.py \
    --method "dad" \
    --num-steps 500 \
    --num-inner-samples 100 \
    --num-outer-samples 200 \
    --seed -1 \
    --mlflow-experiment-name "dad_regression" \
    --num_experiments 10
```

To train a DAD network for toy example under mis-specified model in the paper,

```bash
python3 toy_examples/dad_regression/script/train_dad_regression_linear.py \
    --method "dad" \
    --num-steps 500 \
    --num-inner-samples 100 \
    --num-outer-samples 200 \
    --seed -1 \
    --mlflow-experiment-name "dad_regression_linear" \
    --num_experiments 10
```


The method have "boed", "random","dad", and "boed_new".  and the expeiriment_id and run_id need to fill into the main.py

```bash
python3 toy_examples/dad_regression/script/fixedmodel/main.py
```

To show the degree of covariate shift and generalization error

```bash
python3 toy_examples/dad_regression/script/fixedmodel/plotting.py
```


## Experiment: Location Finding

To train a DAD network for location finding in the paper, the code is adapted from [https://github.com/ae-foster/dad](https://github.com/ae-foster/dad)


The location_method have "boed", "random","dad", and "boed_new".


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
python3 toy_examples/sources_mcmc_vi/script/plotting.py
```

