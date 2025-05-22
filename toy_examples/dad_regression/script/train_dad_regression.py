import argparse
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from torch import nn
import mlflow
import mlflow.pytorch
from tqdm import trange
from pyro.infer.util import torch_item


import sys, os

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation
from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
    RandomDesignBaseline,
    rmv,
)

# -------------------------- Neural Network for Design Selection --------------------------

class EncoderNetwork(nn.Module):
    """Encodes past (x, y) pairs into a latent representation"""
    def __init__(self, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim 
        self.linear1 = nn.Linear(2, hidden_dim)  # Input: (x, y)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.tanh = nn.Tanh()

    def forward(self, xi, y,  **kwargs):
        inputs = torch.cat([xi, y], dim=-1)
        x = self.linear1(inputs)
        x = self.tanh(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    """Outputs the next design (x) given the latent state"""
    def __init__(self, encoding_dim):
        super().__init__()
        self.linear = nn.Linear(encoding_dim, 1)
        # Initialize weight and bias to cover wider range
        nn.init.uniform_(self.linear.weight, a=-0.5, b=0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, latent):
        x = self.linear(latent)
        x = x.squeeze(-1)
        x = torch.tanh(x) *4  
        # print("emitter network",x.min(),x.max)
        return x  # Outputs x



# ------------------------- DAD for Nonlinear Regression ---------------------------------

class NonlinearRegressionExperiment(nn.Module):
    """Deep Adaptive Design (DAD) applied to a nonlinear regression experiment"""
    def __init__(
        self,
        design_net,
        beta_loc=None,
        beta_scale=None,
        noise_scale=0.1,  # Noise in y
        T=30,  # Number of experiments
    ):
        super().__init__()
        self.design_net = design_net
        self.T = T

        # Prior for beta coefficients (Unknown regression parameters)
        if beta_loc is None:
            beta_loc = torch.tensor([-0.5, 0.0, 0.5])  # Prior mean
        if beta_scale is None:
            beta_scale = torch.tensor([1.0, 1.0, 1.0])  # Prior std dev

        self.beta_prior = dist.Normal(beta_loc, beta_scale).to_event(1)
        self.noise_scale = noise_scale

    def forward_map(self, x, beta):
        """Nonlinear regression model: y = β0 + β1x + β2x² + noise x.shape = (200)"""
            # Ensure beta has the correct shape
        beta0 = beta[..., 0]  # Shape: (200)
        beta1 = beta[..., 1]  # Shape: (200)
        beta2 = beta[..., 2]  # Shape: (200)

        return beta0 + beta1 * x + beta2 * (x ** 2)

    def model(self):
        pyro.module("design_net", self.design_net)

        # Sample unknown parameters β0, β1, β2
        beta = latent_sample("beta", self.beta_prior)
        y_outcomes = []
        x_designs = []

        for t in range(self.T):
            # Select design x_t (DAD chooses this)
            xi = compute_design(f"xi{t + 1}", self.design_net.lazy(*zip(x_designs, y_outcomes)))
            # print("xi",xi.min(),xi.max())
            # xi = torch.clamp(xi, min=-4, max=4) 
            # Generate noisy observation y_t
            mean_y = self.forward_map(xi, beta)
            y = observation_sample(f"y{t + 1}", dist.Normal(mean_y, self.noise_scale))

            xi = xi.unsqueeze(-1)
            y = y.unsqueeze(-1)
            y_outcomes.append(y)
            x_designs.append(xi)
        # print("x_designs:", torch.vstack(x_designs).min().item(), torch.vstack(x_designs).max().item())
        return y_outcomes


    def forward(self, beta=None):
        """Run the policy"""
        # self.design_net.eval()
        if beta is not None:
            model = pyro.condition(self.model, data={"beta": beta})
        else:
            model = self.model
        designs = []
        observations = []

        with torch.no_grad():
            trace = pyro.poutine.trace(model).get_trace()
            for t in range(self.T):
                xi = trace.nodes[f"xi{t + 1}"]["value"]
                designs.append(xi)

                y = trace.nodes[f"y{t + 1}"]["value"]
                observations.append(y)
        return torch.cat(designs).unsqueeze(1), torch.cat(observations).unsqueeze(1)



# -------------------------- Train DAD for Nonlinear Regression --------------------------

def train_dad(
    seed,
    mlflow_experiment_name,
    num_steps=500,
    num_inner_samples=100,  
    num_outer_samples=200,  
    lr=0.001,
    gamma=0.95,
    hidden_dim=32,
    encoding_dim=8,
    T=30,  # Number of experiments
    empty_value = 0.0,
    method = "dad",
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    empty_value = torch.tensor(empty_value)
    # Define Encoder & Emitter Networks
    encoder = EncoderNetwork(hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim)

    # DAD Design Network
    design_net = SetEquivariantDesignNetwork(
        encoder, emitter, empty_value=empty_value,device="cpu",   # Start from x = 0
    )

    # Define the Nonlinear Regression Experiment
    model = NonlinearRegressionExperiment(
        design_net=design_net,
        beta_loc=torch.tensor([0.0, 1.0, -0.5]),  # Prior mean
        beta_scale=torch.tensor([1.0, 1.0, 1.0]),  # Prior std dev
        noise_scale=0.1,
        T=T,
    )

    # Define optimizer & loss function
    scheduler = pyro.optim.ExponentialLR(
        {"optimizer": torch.optim.Adam, "optim_args": {"lr": lr}, "gamma": gamma}
    )
    loss_fn = PriorContrastiveEstimation(num_outer_samples, num_inner_samples)
    oed = OED(model.model, scheduler, loss_fn,device="cpu")

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("seed", seed)
    ## Model hyperparams
    mlflow.log_param("num_experiments", T)
    ## Design network hyperparams

    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)

    ## Optimiser hyperparams
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("gamma", gamma)

    loss_history = []
    num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
    # Train DAD to optimize x selection
    for i in num_steps_range:
        model.design_net.train()
        loss = oed.step()
        loss = torch_item(loss)
        loss_history.append(loss)
        print("loss:",loss)
        if (i +1) % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval)
            print("Storing model to MlFlow... ", end="")
            mlflow.pytorch.log_model(model.cpu(), "model")
            ml_info = mlflow.active_run().info
            model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model{str(i)}"
            print(f"Model sotred in {model_loc}. Done.")


        # Decrease LR at every 1K steps
        if i % 500 == 0:
            scheduler.step()

    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    print("Storing model to MlFlow... ", end="")
    # mlflow.pytorch.log_model(model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"./mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model sotred in {model_loc}. Done.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    return model

# -------------------------- Run Experiment and Plot Results --------------------------

def run_experiment(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma,
    T,
    hidden_dim,
    encoding_dim,
    mlflow_experiment_name,
    empty_value,
    method,
):
    # Train DAD for nonlinear regression
    dad_model = train_dad(
        seed= seed,
        mlflow_experiment_name = mlflow_experiment_name,
        num_steps=num_steps,
        num_inner_samples=num_inner_samples,  
        num_outer_samples=num_outer_samples,  
        lr=lr,
        gamma=gamma,
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim,
        T=T, 
        empty_value =empty_value, #
        method= method,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAD for Nonlinear Regression")
    parser.add_argument("--num-steps", default=1000, type=int)
    parser.add_argument("--num-inner-samples", default=100, type=int)
    parser.add_argument("--seed", default=952492480, type=int)
    parser.add_argument("--num-outer-samples", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--hidden-dim", default=32, type=int)
    parser.add_argument("--encoding-dim", default=8, type=int)
    parser.add_argument("--num_experiments", default=10, type=int)  # Number of experiments T
    parser.add_argument("--empty_value", default=0.0, type=int)
    parser.add_argument(
        "--mlflow-experiment-name", default="dad_regression", type=str
    )
    parser.add_argument(
        "--method", default="dad", type=str
    )

    args = parser.parse_args()

    # Run experiment
    run_experiment(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        mlflow_experiment_name=args.mlflow_experiment_name,
        empty_value = args.empty_value,
        method = "dad",)