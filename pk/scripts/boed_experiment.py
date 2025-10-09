# boed_experiment.py
import os
import torch
import pyro
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

from plotting import compute_mmd_dispersion 

from tqdm import tqdm
import math


def fit_g_mle_with_margin_like_model(xis, ys, model, 
                                     tau=2.0, alpha=1.0, 
                                     steps=200, lr=1e-2, l2=0.0, 
                                     init="prior"):
    if xis.numel() == 0:
        
        def g_mean_fn(x):
            return model.forward_map(x, model.theta_loc)
        return g_mean_fn

    x = xis.view(-1, model.p)
    y = ys.view(-1, 1)

    with torch.no_grad():
        mu_f = model.forward_map(x, model.theta_loc).detach()  # [n,1]

    if init == "zeros":
        theta_init = torch.zeros_like(model.theta_loc)
    else:
        theta_init = model.theta_loc.clone()



    theta_g = torch.nn.Parameter(theta_init)
    opt = torch.optim.Adam([theta_g], lr=lr)
    print("start adversial training!")
    pbar = tqdm(range(steps), desc="Training", leave=True)
    for step in pbar:
        g_vals = model.forward_map(x, theta_g)           # [n,1]
        mse = torch.mean((g_vals - y) ** 2)
        margin = torch.relu(tau - (mu_f - g_vals).abs()).mean()
        reg = l2 * (theta_g ** 2).mean()
        loss = mse + alpha * margin + reg

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 ==0:
            print(f"Step {step}: loss = {loss.item():.6f}")
    theta_g_f = theta_g.detach().clone()

    def g_mean_fn(x_new):
        return model.forward_map(x_new, theta_g_f)       
    return g_mean_fn 



def _disagreement_weight(mu_f, mu_g, tau=None, smooth_kappa=None):
    """

    """
    gap = (mu_f - mu_g).abs().squeeze(-1)
    if tau is None:
        tau = torch.quantile(gap, 0.8)
    if smooth_kappa is None:
        return (gap >= tau).float()
    else:
        return torch.sigmoid((gap - tau) / smooth_kappa)
    
class BOEDExperiment:
    def __init__(
        self,
        model,
        design_space=(0.0, 24.0),      # (t_min, t_max) 也可以传 (t_min, t_max, step)
        num_candidates=100,
        eig_method="nmc_eig",
        dir="",
        biased_sample_flag=False,
    ):
        self.model = model
        self.num_candidates = num_candidates
        self.biased_sample_flag = biased_sample_flag
        self.design_space = design_space
        self.candidate_designs = self.initialize_candidate_designs(self.num_candidates)
        self.eig_method = eig_method
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def initialize_candidate_designs(self, num_candidates, seed=777):
        #  \xi \in [t_min, t_max]，
        if len(self.design_space) == 3:
            t_min, t_max, t_step = self.design_space
            grid = torch.arange(t_min, t_max + 1e-8, t_step).unsqueeze(-1)  # [N,1]
        else:
            t_min, t_max = self.design_space
            grid = torch.arange(t_min, t_max + 1e-8, 0.1).unsqueeze(-1)

        # local seed
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if num_candidates >= len(grid):
                candidate_designs = grid
            else:
                idx = torch.randperm(len(grid))[:num_candidates]
                candidate_designs = grid[idx]

        return candidate_designs  # [N,1]

    def optimize_design(self, xis, ys, current_model, alg_method="boed", db_args=None):
        optimizer = pyro.optim.Adam({"lr": 0.001, "weight_decay": 1e-4})
        indices = torch.randperm(self.candidate_designs.size(0))
        candidate_designs = self.candidate_designs[indices]

        #  EIG
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
        else:
            raise ValueError(f"Unknown eig_method: {self.eig_method}")

        # RI
        if alg_method.startswith("boed_new"):
            if xis.numel() == 0:
                div_ori = torch.tensor(1e+6, dtype=torch.float32)
            else:
                div_ori = compute_mmd_dispersion(xis)
            new_designs = torch.cat([xis.reshape(-1).repeat(candidate_designs.size(0), 1), candidate_designs], dim=1)
            div = torch.tensor([compute_mmd_dispersion(d) for d in new_designs]) / div_ori
            if alg_method == "boed_new_025":
                eig = eig * (1 - 0.25 * div)
            elif alg_method == "boed_new_05":
                eig = eig * (1 - 0.5 * div)
            else:
                eig = eig * (1 - div)

        # === RIDeA===
        if alg_method.startswith("db_boed"):
            db_args = db_args or {}
            tau = db_args.get("tau", None)
            alpha = db_args.get("alpha", 1.0)
            steps = db_args.get("steps", 200)
            lr = db_args.get("lr", 1e-2)
            l2 = db_args.get("l2", 0.0)
            init = db_args.get("init", "prior")
            smooth_kappa = db_args.get("smooth_kappa", 0.25)
            mmd_lambda = db_args.get("lambda", 0.0)

            g_mean_fn = fit_g_mle_with_margin_like_model(
                xis, ys, current_model,
                tau=(tau if isinstance(tau, (float, int)) else 10.0),
                alpha=alpha, steps=steps, lr=lr, l2=l2, init=init
            )

            with torch.no_grad():
                mu_f_cand = current_model.forward_map(candidate_designs, current_model.theta_loc)
                mu_g_cand = g_mean_fn(candidate_designs)
                W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau, smooth_kappa=smooth_kappa)
                if torch.all(W == 0):
                    gap = (mu_f_cand - mu_g_cand).abs().squeeze(-1)
                    tau_backoff = torch.quantile(gap, 0.8) if tau is None else max(1e-6, float(tau) * 0.5)
                    W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau_backoff, smooth_kappa=smooth_kappa)

                eig = eig * (W + (W == 0).float() * 0.05)

                if mmd_lambda > 0.0 and xis.numel() > 0:
                    div_ori = compute_mmd_dispersion(xis)
                    new_designs = torch.cat([xis.reshape(-1).repeat(candidate_designs.size(0), 1),
                                             candidate_designs], dim=1)
                    div = torch.tensor([compute_mmd_dispersion(d) for d in new_designs]) / div_ori
                    eig = eig * (1 - mmd_lambda * div)

        best_idx = torch.argmax(eig)
        return candidate_designs[best_idx]

    def run_svi(self, xis, ys):
        #
        pyro.clear_param_store()
        self.model.reset()

        optimizer = pyro.optim.Adam({"lr": 0.001})

        # insert observation
        conditioned_model = pyro.condition(self.model.make_model(), data={"y": ys})

        #  (model, guide, optim, loss)
        svi = SVI(conditioned_model, self.model.guide, optimizer, loss=Trace_ELBO(num_particles=2))

        # iteration
        for _ in range(5000):
            svi.step(xis)

        # 
        store = pyro.get_param_store()
        if "posterior_loc" in store and "posterior_scale_tril" in store:
            loc = store["posterior_loc"].detach()
            L = store["posterior_scale_tril"].detach()
            #
            cov = L @ L.transpose(-1, -2)   
            return loc, cov

        # 
        if "posterior_mean" in store and "posterior_sd" in store:
            mean = store["posterior_mean"].detach()
            sd_or_cov = store["posterior_sd"].detach()
            if sd_or_cov.ndim == 1:
                # 
                cov = torch.diag(sd_or_cov ** 2)
            else:
                # 
                cov = 0.5 * (sd_or_cov + sd_or_cov.mT)  #
            return mean, cov

        raise RuntimeError(
            "SVI finished but no posterior parameters found. "
            "Expected ('posterior_loc','posterior_scale_tril') or ('posterior_mean','posterior_sd')."
        )

    def run_mcmc(self, xis, ys):
        pyro.clear_param_store()
        self.model.reset()

        conditioned_model = pyro.condition(self.model.make_model(), data={"y": ys})
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=1000, num_chains=1)
        mcmc.run(xis)

        samples = mcmc.get_samples()["theta"]  # [S, 3]
        mean = samples.mean(dim=0)

        # 
        centered = samples - mean
        cov = centered.mT @ centered / (samples.shape[0] - 1)

        return mean.detach(), cov.detach()