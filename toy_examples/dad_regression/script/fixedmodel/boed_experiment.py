import torch
import pyro
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import MCMC, NUTS
import os
from utils import compute_mmd_dispersion

class BOEDExperiment:
    def __init__(self, model, design_space=(-4, 4), num_candidates=100, eig_method="marginal_eig", dir = "./test/"):
        self.model = model
        self.candidate_designs = torch.linspace(*design_space, steps=num_candidates).unsqueeze(-1)
        self.eig_method = eig_method
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
    def optimize_design(self, xis, ys, current_model, LAMBDA = -1):
        """Optimize the design using EIG."""
        # pyro.clear_param_store()
        optimizer = Adam({"lr": 0.01})
        indices = torch.randperm(self.candidate_designs.size(0))  # Generate random permutation of indices

        self.candidate_designs = self.candidate_designs[indices] 

        # print(self.candidate_designs)
        if self.eig_method == "marginal_eig":
            eig = marginal_eig(
                current_model.make_model(),
                self.candidate_designs,
                "y",
                "beta",
                num_samples=100,
                guide=current_model.marginal_guide,
                optim=optimizer,
                num_steps=100,
                final_num_samples=500
            )
        else:
            eig = nmc_eig(
                current_model.make_model(),
                self.candidate_designs,
                observation_labels="y",
                target_labels="beta",
                N=300,  # Outer samples 300
                M=30   # Inner samples 30
            )
        if LAMBDA != -1:
            # compute original Wasserstein dispersion
            if xis.numel() == 0:
                div_ori = torch.tensor(1e+6, dtype=torch.float32)
            else:
                div_ori = compute_mmd_dispersion(xis)

            # compute the mmd dispersion changed
            new_designs = torch.cat([xis.repeat(self.candidate_designs.size(0), 1), self.candidate_designs], dim=1)
            div_new = torch.tensor([compute_mmd_dispersion(design) for design in new_designs])
            div = div_new / div_ori


            print("*********************")
            # print("div_new", div_new)
            # print("div_ori", div_ori)
            # print("div", div)
            # print("1-div", 1-div)
            # print("eig", eig)
            eig = eig * (1 - LAMBDA * div)
            
                


        best_design_idx = torch.argmax(eig)
        print(best_design_idx)
        best_design = self.candidate_designs[best_design_idx]
        print(f"Best design at round: {best_design.item()}")
        return best_design, eig

    def run_svi(self, xis, ys):
        """Run SVI to infer posterior."""
        self.model.reset()
        conditioned_model = pyro.condition(self.model.make_model(), {"y": ys})
        svi = SVI(conditioned_model, self.model.guide, Adam({"lr": 0.005}), Trace_ELBO())
        
        for _ in range(2000):  # Optimization loop
            svi.step(xis)
        
        posterior_mean = pyro.param("posterior_mean").detach()
        posterior_sd = pyro.param("posterior_sd").detach()
        return posterior_mean, posterior_sd
    
    def run_mcmc(self, xis, ys):
        self.model.reset()
        conditioned_model = pyro.condition(self.model.make_model(), data={"y": ys})
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=200, num_chains=1)
        mcmc.run(xis)
        # Extract samples
        posterior_samples = mcmc.get_samples()
        beta = posterior_samples["beta"]

        posterior_mean = beta.mean(dim =0)
        posterior_sig = beta.std(dim=0)

        return posterior_mean,posterior_sig+ 1e-5
