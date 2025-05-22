import os
import torch
import pyro
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from plotting import compute_mmd_dispersion


class BOEDExperiment:
    def __init__(self, model, design_space=(-4, 4), num_candidates=100, eig_method="marginal_eig", dir="", biased_sample_flag=False):
        self.model = model
        self.num_candidates = num_candidates
        self.biased_sample_flag = biased_sample_flag
        self.candidate_designs = self.initialize_candidate_designs(self.num_candidates)
        self.eig_method = eig_method
        self.dir = dir

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def initialize_candidate_designs(self, num_candidates, seed=777):
        ranges = [torch.arange(-4, 4.1, 0.1) for _ in range(self.model.p)]
        grid = torch.cartesian_prod(*ranges)
        torch.manual_seed(seed)
        indices = torch.randperm(len(grid))[:num_candidates]
        candidate_designs = grid[indices]
        return candidate_designs

    def optimize_design(self, xis, ys, current_model,alg_method = "boed"):
        optimizer = pyro.optim.Adam({"lr": 0.001, "weight_decay": 1e-4})
        indices = torch.randperm(self.candidate_designs.size(0))
        candidate_designs = self.candidate_designs[indices]

        if self.eig_method == "marginal_eig":
            eig = marginal_eig(
                current_model.make_model(),
                candidate_designs,
                "y",
                "theta",
                num_samples=1000,
                guide=current_model.marginal_guide,
                optim=optimizer,
                num_steps=100,
                final_num_samples=10000
            )
        elif self.eig_method == "nmc_eig":
            eig = nmc_eig(
                current_model.make_model(),
                candidate_designs,
                observation_labels="y",
                target_labels="theta",
                N=300,
                M=30
            )
        if alg_method == "boed_new":
            if xis.numel() == 0:
                div_ori = torch.tensor(1e+6, dtype=torch.float32)
            else:
                div_ori = compute_mmd_dispersion(xis)

            # compute the mmd dispersion changed
            new_designs = torch.cat([xis.reshape(-1).repeat(self.candidate_designs.size(0), 1), self.candidate_designs], dim=1)
            div = torch.tensor([compute_mmd_dispersion(design) for design in new_designs]) / div_ori
            eig = eig * (1- div)

        best_design_idx = torch.argmax(eig)
        best_design = candidate_designs[best_design_idx]
        return best_design

    def run_svi(self, xis, ys):
        self.model.reset()
        optimizer = pyro.optim.Adam({"lr": 0.001})
        conditioned_model = pyro.condition(self.model.make_model(), {"y": ys})
        svi = SVI(conditioned_model, self.model.guide, optimizer, Trace_ELBO(num_particles=2), num_samples=1000)
        
        best_loss = float("inf")
        patience_counter = 0
        for step in range(5000):
            loss = svi.step(xis)
        posterior_mean = pyro.param("posterior_mean").detach()
        posterior_sd = pyro.param("posterior_sd").detach()
        return posterior_mean, posterior_sd

    def run_mcmc(self, xis, ys):
        self.model.reset()
        conditioned_model = pyro.condition(self.model.make_model(), data={"y": ys})
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=1000, num_chains=1)
        mcmc.run(xis)
        posterior_samples = mcmc.get_samples()
        beta = posterior_samples["theta"]
        beta = beta.mean(dim=1)
        posterior_mean = beta.mean(dim=0)
        posterior_sig = torch.diag_embed(beta.std(dim=0))
        return posterior_mean, posterior_sig
