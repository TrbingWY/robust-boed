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
        self.design_space = design_space
        self.candidate_designs = torch.linspace(*design_space, steps=num_candidates).unsqueeze(-1)
        self.eig_method = eig_method
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
    def optimize_design(self, xis, ys, current_model, LAMBDA=-1, db_args=None):
        """db_args: None or dict;Optimize the design using EIG."""
        # pyro.clear_param_store()
        optimizer = Adam({"lr": 0.01})
        indices = torch.randperm(self.candidate_designs.size(0))  # Generate random permutation of indices
        # indices = torch.tensor([20, 42,  9, 54, 51, 31, 41, 18, 40, 12, 25,  6, 39,  2, 55, 23, 28, 14,
        # 52, 19, 36, 10, 22, 57, 59, 35,  5, 24,  4, 29, 50,  8,  7, 26, 48,  1, 0, 11, 21, 17, 46, 49, 34, 56, 30, 44, 53, 33, 37, 58, 27, 47, 38, 32, 13, 45, 16,  3, 15, 43])

        # self.candidate_designs = torch.linspace(-4, 4, steps=17).unsqueeze(-1)
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
            print("LAMBDA Process!!")
            if xis.numel() == 0:
                div_ori = torch.tensor(1e+6, dtype=torch.float32)
            else:
                div_ori = compute_mmd_dispersion(xis)

            # compute the mmd dispersion changed
            if xis.ndim > 1:
                xis= xis.squeeze(-1)
            new_designs = torch.cat([xis.repeat(self.candidate_designs.size(0), 1), self.candidate_designs], dim=1)
            div_new = torch.tensor([compute_mmd_dispersion(design) for design in new_designs])
            div = div_new / div_ori

          
            eig = eig * (1 - LAMBDA * div)
           
        
        if db_args is not None:
            print("db_args Process!!")
            tau = db_args.get("tau", 0.25)
            lengthscale = db_args.get("lengthscale", 0.8)
            lam = db_args.get("lam", 1e-2)
            smooth_kappa = db_args.get("smooth_kappa", True)
            TEST = db_args.get("test", False)
            alpha = db_args.get("alpha", 1.0)
            mmd_lambda   = db_args.get("lambda", 1.0) 

           
            
             ###################################################Attention!!!!!
            if TEST:
                g_mean_fn = _make_g_from_beta(current_model, current_model.BEST_BETA)

            else:
           
                _disagreement_xis = torch.linspace(*self.design_space, steps=200).unsqueeze(-1)

                g_mean_fn = fit_g_mle_with_margin(xis=xis,ys=ys, model=current_model,tau=2.0,alpha=alpha)
        
            with torch.no_grad():
                mu_f_cand = _poly_predict_mean(current_model, self.candidate_designs)  # [N,1]

                mu_g_cand = g_mean_fn(self.candidate_designs)  
                # mu_g_cand = mu_f_cand + torch.randn_like(mu_f_cand) * 3                        # [N,1]

            # 3) compute weigh
            W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau, smooth_kappa=smooth_kappa).squeeze(-1)  # [N]

            
            # print("W.shape", W.shape)
            # print("eig.shape",eig.shape)

            # （optional）if number of candidates designs is zero
            if torch.all(W == 0):
                print("=**************back************")

                #back strategy
                tau_backoff = max(1e-6, tau * 0.5)
                W = _disagreement_weight(mu_f_cand, mu_g_cand, tau=tau_backoff, smooth_kappa=smooth_kappa).squeeze(-1)
                print("W",W)
                eig = eig * (W + (W == 0).float() * 0.1)  
            else:
               
                eig = eig * W


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
        # posterior_sd = pyro.param("posterior_sd").detach()
        # return posterior_mean, posterior_sd
        L = pyro.param("posterior_scale_tril").detach()   # 下三角
        posterior_cov = L @ L.T                           # 协方差矩阵
        return posterior_mean, posterior_cov
    
    def run_mcmc(self, xis, ys):
        pyro.clear_param_store()
        self.model.reset()
        conditioned_model = pyro.condition(self.model.make_model(), data={"y": ys})
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=200, num_chains=1)
        mcmc.run(xis)
        # Extract samples
        posterior_samples = mcmc.get_samples()
        beta = posterior_samples["beta"]
        posterior_mean = beta.mean(dim =0)
        # L = beta.std(dim =0)
        # posterior_cov = L @ L.T
        # posterior_cov = beta.std(dim =0)
        # --- ---
        X = beta - posterior_mean
        n = X.size(0)
        cov = (X.T @ X) / (n - 1)  

        return posterior_mean,cov




def _disagreement_weight(mu_f, mu_g, tau, smooth_kappa=None):
    """
    W_{f,g}(x) = 1{|mu_f - mu_g| >= tau}
    
    """
    if tau is None:
        tau = torch.quantile((mu_f - mu_g).abs(), 4/5)
        # tau = (mu_f - mu_g).abs().max()/1.5
        # print("tau.....",tau)

    gap = (mu_f - mu_g).abs()
    # print("gap",gap)
    if smooth_kappa is None:
        # print((gap >= tau).float())
        return (gap >= tau).float()
    else:
        # sigma((gap - tau)/kappa)
        # print("sigmoid")
        return torch.sigmoid((gap - tau) / smooth_kappa)
    
def _make_g_from_beta(model, beta):
    """
    return g_mean_fn(x) = f_true(x) = sum beta[i] x^i
    """
    # beta = beta.to(dtype=model.prior_mean.dtype, device=model.prior_mean.device)
    def g_mean(x):
        return _poly_predict_from_beta(beta, x)
    return g_mean



def _poly_predict_mean(model, x):  # x: [N,1] or [1,1]
    
    pm = model.prior_mean  # shape [degree+1]
    deg = model.degree
    # sum pm[i] x^i
    mu = torch.zeros_like(x.squeeze(-1))
    for i in range(deg + 1):
        mu = mu + pm[i] * (x.squeeze(-1) ** i)
    return mu.unsqueeze(-1)  # [N,1]

def _poly_predict_from_beta(beta, x):
    """
    beta: shape [degree+1]
    x: [N,1] or [1,1]
    return: [N,1]
    """
    x = x.view(-1, 1)
    deg = beta.shape[0] - 1
    mu = torch.zeros_like(x.squeeze(-1))
    for i in range(deg + 1):
        mu = mu + beta[i] * (x.squeeze(-1) ** i)
    return mu.unsqueeze(-1)


def fit_g_mle_with_margin(xis, ys, model, tau=2.0, alpha=1.0, steps=200, lr=1e-2):
    x = xis.reshape(-1)
    y = ys.reshape(-1)
    with torch.no_grad():
        mu_f = _poly_predict_mean(model, xis.reshape(-1,1)).reshape(-1)  # [N]

    #  g
    b0 = torch.zeros(1, device=x.device, dtype=x.dtype, requires_grad=True)
    b1 = torch.zeros(1, device=x.device, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.Adam([b0, b1], lr=lr)

    for _ in range(steps):
        g_vals = b0 + b1 * x
        mse = torch.mean((g_vals - y)**2)
        margin = torch.relu(tau - (mu_f - g_vals).abs()).mean()
        loss = mse + alpha * margin
        opt.zero_grad()
        loss.backward()
        opt.step()

    beta = torch.stack([b0.detach().squeeze(), b1.detach().squeeze()])
    def g_mean_fn(x_tensor):
        xt = x_tensor.reshape(-1)
        return (beta[0] + beta[1]*xt).reshape(-1,1)
    return g_mean_fn
