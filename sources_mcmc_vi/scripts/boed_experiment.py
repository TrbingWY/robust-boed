import os
import torch
import pyro
import sys
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from plotting import compute_mmd_dispersion

import math
# from epig.epig_logprobs import epig_from_logprobs
# from epig.epig_regression import epig_regression
from tqdm import tqdm


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

    def optimize_design(self, xis, ys, current_model, alg_method="boed", db_args=None):
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
        if alg_method.startswith("boed_new"):
            if xis.numel() == 0:
                div_ori = torch.tensor(1e+6, dtype=torch.float32)
            else:
                div_ori = compute_mmd_dispersion(xis)

            # compute the mmd dispersion changed
            new_designs = torch.cat([xis.reshape(-1).repeat(candidate_designs.size(0), 1), candidate_designs], dim=1)
            div = torch.tensor([compute_mmd_dispersion(design) for design in new_designs]) / div_ori
            if alg_method == "boed_new_025":
                eig = eig * (1- 0.25 * div)
            elif alg_method == "boed_new_05":
                eig = eig * (1- 0.5 * div)
            else:
                eig = eig * (1- div)

                # === （Disagreement-Based BOED）===
                # === ：db_boed（g has the same structure with SourceModel ===
        if alg_method.startswith("db_boed"):
            db_args = db_args or {}
            tau          = db_args.get("tau", None)          # None represent the thresohold 
            alpha        = db_args.get("alpha", 1.0)
            steps        = db_args.get("steps", 200)
            lr           = db_args.get("lr", 1e-2)
            l2           = db_args.get("l2", 0.0)
            init         = db_args.get("init", "prior")
            smooth_kappa = db_args.get("smooth_kappa", 0.25)
            mmd_lambda   = db_args.get("lambda", 0.0)    # when >0,add the MMD penalty

            # print("tau......",tau)
            # 1) g has the same model structure with thef: g：g(x)=forward_map(x, theta_g)
            g_mean_fn = fit_g_mle_with_margin_like_model(
                xis, ys, current_model, 
                tau=(tau if isinstance(tau, (float,int)) else 10.0),  # set tau
                alpha=alpha, steps=steps, lr=lr, l2=l2, init=init
            )

            with torch.no_grad():
                # 2) compute mu_f and mu_g among candidate designs
                mu_f_cand = current_model.forward_map(candidate_designs, current_model.theta_loc)  # [N,1]
                mu_g_cand = g_mean_fn(candidate_designs)                                          # [N,1]

                # 3) disagreement weigh(Here, tau can be  None or automatic）
                W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau, smooth_kappa=smooth_kappa)  # [N]

                if torch.all(W == 0):
                    #
                    gap = (mu_f_cand - mu_g_cand).abs().squeeze(-1)
                    tau_backoff = torch.quantile(gap, 0.8) if tau is None else max(1e-6, float(tau)*0.5)
                    W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau_backoff, smooth_kappa=smooth_kappa)

                eig = eig * (W + (W == 0).float() * 0.05)  # 

                # (optional) MMD penalty（similar to the boed_new_025 method）
                if mmd_lambda > 0.0 and xis.numel() > 0:
                    div_ori = compute_mmd_dispersion(xis)
                    new_designs = torch.cat([
                        xis.reshape(-1).repeat(candidate_designs.size(0), 1),
                        candidate_designs
                    ], dim=1)
                    div = torch.tensor([compute_mmd_dispersion(d) for d in new_designs]) / div_ori
                    eig = eig * (1 - mmd_lambda * div)
        
        best_design_idx = torch.argmax(eig)
        best_design = candidate_designs[best_design_idx]
        return best_design
    
    # def select_via_epig(self, model,test_x, K=10, num_targets=100):
    #     # candidate_xs = self.candidate_designs  # shape: [N_pool, D]
        
    #     # 2. build target designs (test_x*)
    #     indices = torch.randperm(self.candidate_designs.size(0))
    #     candidate_xs = self.candidate_designs[indices]
    #     target_xs = test_x

    #     # 3. sampling theta ~ posterior, predict p(y|x, theta)
    #     theta_samples = torch.distributions.MultivariateNormal(
    #         model.theta_loc, model.theta_covmat
    #     ).sample(sample_shape=(K,))  # [K, K, D]

    #     def predict_log_prob(xs, theta_samples):
    #         # xs: [N, D]
    #         # theta_samples: [K, K, D]
    #         preds = []
    #         for theta in theta_samples:
    #             y_mu = model.forward_map(xs, theta)  # [N, 1]
    #             preds.append(y_mu.squeeze(-1))  # [N]
    #         return torch.stack(preds, dim=1)  # [N, K]

    #     pool_log_probs = predict_log_prob(candidate_xs, theta_samples)  # [N_pool, K]
    #     targ_log_probs = predict_log_prob(target_xs, theta_samples)     # [N_targ, K]

    #     # 4. call EPIG score function（definition can be found in the next step）
    #     epig_scores = epig_regression(pool_log_probs, targ_log_probs)

    #     # print("epig_scores.shape:",epig_scores.shape)
    #     # 5. return the best design
    #     best_idx = torch.argmax(epig_scores)
    #     return candidate_xs[best_idx]

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




def _disagreement_weight(mu_f, mu_g, tau=None, smooth_kappa=None):
    """
    mu_f, mu_g: [N, 1]
    Returns W: [N].
    When tau=None, an adaptive threshold is used, set to the 80th percentile of |mu_f - mu_g|.
    If smooth_kappa=None, a hard threshold is applied; otherwise, a smooth weighting
    is used with sigmoid((gap - tau) / smooth_kappa).
    """
    gap = (mu_f - mu_g).abs().squeeze(-1)
    if tau is None:
        tau = torch.quantile(gap, 0.8)
    if smooth_kappa is None:
        return (gap >= tau).float()
    else:
        return torch.sigmoid((gap - tau) / smooth_kappa)

import torch

def fit_g_mle_with_margin_like_model(xis, ys, model, 
                                     tau=2.0, alpha=1.0, 
                                     steps=200, lr=1e-2, l2=0.0, 
                                     init="prior"):
    """
    Fit an adversarial model g using the same `forward_map` structure as SourceModel:
        g(x) = forward_map(x, theta_g)
    where theta_g is obtained by minimizing
        MSE(g(x_i), y_i) 
        + alpha * E[max(0, tau - |mu_f(x_i) - g(x_i)|)] 
        + l2 * ||theta_g||^2
    Here, mu_f(x) = forward_map(x, theta_loc) is the mean of the current model f.

    Args:
        xis: [n, p]
            Input data.
        ys: [n, 1] or [n]
            Target values.
        model: SourceModel or SourceModel_Mis
            Must have attributes `.forward_map`, `.theta_loc`, `.p`, and `.K`.
        tau: float
            Margin threshold.
        alpha: float
            Margin penalty weight.
        steps: int
            Number of optimization steps.
        lr: float
            Learning rate.
        l2: float, optional
            L2 regularization coefficient for theta_g.
        init: str
            Initialization method for theta_g:
                - "prior": initialize from current theta_loc
                - "zeros": initialize with all zeros

    Returns:
        g_mean_fn(x: [N, p] -> [N, 1]):
            A callable function that computes mu_g(x) for candidate points.
    """
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

 
        # pbar.set_postfix({
        #     "loss": f"{loss.item():.6f}",
        #     "mse": f"{mse.item():.6f}",
        #     "margin": f"{margin.item():.6f}",
        #     "reg": f"{reg.item():.6f}"
        # })
        if step % 100 ==0:
            print(f"Step {step}: loss = {loss.item():.6f}")
    theta_g_f = theta_g.detach().clone()

    def g_mean_fn(x_new):
        return model.forward_map(x_new, theta_g_f)       
    return g_mean_fn 

