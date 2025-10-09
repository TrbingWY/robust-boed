# pk_model.py
import math
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from torch import nn
from pyro import poutine
from pyro.infer import Predictive


class PKModel:

    def __init__(
        self,
        DV: float = 400.0,                  # D_V
        theta_loc: torch.Tensor = None,     #  [3]
        theta_covmat: torch.Tensor = None,  # [3,3]
        p: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.DV = float(DV)
        self.p = int(p)
        self._theta_dim = 3

        device = device if device is not None else torch.device("cpu")

        if theta_loc is None:
            theta_loc = torch.tensor([math.log(1.0), math.log(0.1), math.log(20.0)], dtype=dtype, device=device)
        if theta_covmat is None:
            theta_covmat = torch.diag(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device=device))

        self.theta_loc = theta_loc.clone().to(device=device, dtype=dtype)        # log-space
        self.theta_covmat = theta_covmat.clone().to(device=device, dtype=dtype)  # log-space
        self.set_theta_prior()

    # ---- housekeeping -----------------------------------------------------
    def set_theta_prior(self):
        # MultivariateNormal 
        self.theta_prior = dist.MultivariateNormal(self.theta_loc, self.theta_covmat)

    def reset(self):

        device = self.theta_loc.device
        dtype = self.theta_loc.dtype
        self.theta_loc = torch.tensor([math.log(1.0), math.log(0.1), math.log(20.0)], dtype=dtype, device=device)
        self.theta_covmat = torch.diag(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device=device))
        self.set_theta_prior()

    # ---- model pieces -----------------------------------------------------
    @staticmethod
    def _softplus_eps(x, eps=1e-8):
       
        return torch.sign(x) * torch.log1p(torch.exp(torch.abs(x))) + eps

    def forward_map(self, xi: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
        """
        comput z(\xi; theta)
        input:
            xi:         [..., 1]   design (hour)
            theta_log:  [..., 3]   (log k_a, log k_e, log V)
        output:
            z:          [..., 1]
        """
        xi = xi.view(*xi.shape[:-1], 1)  # [..., 1]
        ka, ke, V = torch.exp(theta_log[..., 0]), torch.exp(theta_log[..., 1]), torch.exp(theta_log[..., 2])
        denom = self._softplus_eps(ka - ke)
        z = (self.DV / V) * (ka / denom) * (torch.exp(-ke * xi) - torch.exp(-ka * xi))
        return z  # [..., 1]

    def make_model(self):
   
        def model(xi):
            theta_log = pyro.sample("theta", self.theta_prior)  # event_shape=(3,)
            with pyro.plate_stack("plates", xi.shape[:-1]):     # 
                mean = self.forward_map(xi, theta_log)          # [..., 1]
                var = 0.01 * (mean ** 2) + 0.1
                sd = torch.sqrt(var.clamp(min=1e-12))
                pyro.sample("y", dist.Normal(mean, sd).to_event(1))
                pyro.deterministic("mean", mean)                # 
        return model

    # ---- guides（供 SVI 使用）---------------------------------------------
    def marginal_guide(self, design, observation_labels="y"):
      
        event_size = design.shape[-1]  # 1
        q_loc = pyro.param(
            "q_y_loc",
            torch.zeros(*design.shape[:-1], event_size, device=design.device, dtype=design.dtype),
        )
        q_scale = pyro.param(
            "q_y_scale",
            torch.ones_like(q_loc),
            constraint=constraints.positive,
        )
        pyro.sample(observation_labels, dist.Normal(loc=q_loc, scale=q_scale).to_event(1))

    def guide(self, xi):
        loc_init = self.theta_loc.detach().clone()
        L_init = torch.linalg.cholesky(self.theta_covmat.detach().clone())

        posterior_loc = pyro.param("posterior_loc", loc_init)
        posterior_scale_tril = pyro.param(
            "posterior_scale_tril", L_init, constraint=constraints.lower_cholesky
        )
        pyro.sample(
            "theta",
            dist.MultivariateNormal(posterior_loc, scale_tril=posterior_scale_tril)
        )

    # ---- utilities --------------------------------------------------------
    def synthetic_data(self, design: torch.Tensor, true_theta_log: torch.Tensor) -> torch.Tensor:
        
        conditioned = pyro.condition(self.make_model(), data={"theta": true_theta_log})
        predictive = Predictive(conditioned, return_sites=["y"], num_samples=1)
        with torch.no_grad():
            out = predictive(design)
            y = out["y"]  # shape: design.shape[:-1] + (1,) y.shape torch.Size([1, 200, 1]) design.shape torch.Size([200, 1])
        return y.detach().clone()

    def predict_mean(self, design: torch.Tensor) -> torch.Tensor:
       
        with torch.no_grad():
            return self.forward_map(design, self.theta_loc.detach())

    def predict_data(self, design: torch.Tensor, theta_log: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        
        if sample_noise:
            conditioned = pyro.condition(self.make_model(), data={"theta": theta_log})
            predictive = Predictive(conditioned, return_sites=["y"],num_samples=1)
            with torch.no_grad():
                out = predictive(design)
                y = out["y"]
            return y.detach().clone()
        else:
            with torch.no_grad():
                mean = self.forward_map(design, theta_log)
            return mean.detach().clone()

    def compute_rmse(self, model, true_theta_log: torch.Tensor, num_mc: int = 10000) -> torch.Tensor:
       
        mean, cov = model.theta_loc, model.theta_covmat
        mvn = dist.MultivariateNormal(mean, covariance_matrix=cov)

        samples = mvn.sample((num_mc,))  # [num_mc, 3]
        rmse = torch.sqrt(torch.mean((samples - true_theta_log) ** 2, dim=-1)).mean()
        print("RMSE (log-θ space):", rmse.item())
        return rmse


# ---------------- Misspecified model ----------------
class PKModel_Mis(PKModel):

    def __init__(
        self,
        DV: float = 400.0,
        theta_loc: torch.Tensor = None,
        theta_covmat: torch.Tensor = None,
        mis_type: str = "dual_abs",             # "lag", "dual_abs", "lag+error_scale"
        t_lag: float = 5,                # hour
        eps_scale_real: float = 0.1,      
        nu_scale_real: float = math.sqrt(0.1), 
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
     
        rho_ka: float = 0.25,            #
        f_fast: float = 0.6,             # 

        df_real: float = 5.0,
    ):
        super().__init__(DV=DV, theta_loc=theta_loc, theta_covmat=theta_covmat, p=1, device=device, dtype=dtype)
        self.mis_type = mis_type
        self.t_lag = float(t_lag)
        self.eps_scale_real = float(eps_scale_real)
        self.nu_scale_real = float(nu_scale_real)
        self.rho_ka = float(rho_ka)
        self.f_fast = float(f_fast)
        self.df_real = float(df_real)

    # ===== real forward====
    def _forward_map_real(self, xi: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
        xi = xi.view(*xi.shape[:-1], 1)
        ka, ke, V = torch.exp(theta_log[..., 0]), torch.exp(theta_log[..., 1]), torch.exp(theta_log[..., 2])
        denom = self._softplus_eps(ka - ke)
        z = (self.DV / V) * (ka / denom) * (torch.exp(-ke * xi) - torch.exp(-ka * xi))
        return z  # [..., 1]

  
    def forward_map(self, xi: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
        """
        t: [..., 1]
        theta_log: [..., 3]  log-parameters
        """
       

        xi = xi.view(*xi.shape[:-1], 1)  # [...,1]

       
        ka1 = torch.exp(theta_log[..., 0])  # 
        ke  = torch.exp(theta_log[..., 1])
        V   = torch.exp(theta_log[..., 2])

    
        rho = torch.tensor(self.rho_ka, dtype=ka1.dtype, device=ka1.device)
        ka2 = torch.clamp(rho, min=1e-6) * ka1
        f   = torch.clamp(torch.tensor(self.f_fast, device=xi.device, dtype=xi.dtype), 1e-6, 1-1e-6)

        denom1 = self._softplus_eps(ka1 - ke)
        denom2 = self._softplus_eps(ka2 - ke)

        term1 = f   * (ka1 / denom1) * (torch.exp(-ke * xi) + torch.exp(-ka1 * xi))
        term2 = (1-f) * (ka2 / denom2) * (torch.exp(-ke * xi) + torch.exp(-ka2 * xi))
        z = ((self.DV-100) / V) * (term1 + term2)
       
        return z  # [...,1]

    def make_model(self):
       
        def model(xi):
            
            theta_log = pyro.sample("theta", self.theta_prior)  # event_shape=(3,)
            with pyro.plate_stack("plates", xi.shape[:-1]):     #
                mean = self.forward_map(xi, theta_log)          # [..., 1]
                # var = 0.02 * (mean ** 2) + 0.2
                var = torch.tensor(0.2)
                sd = torch.sqrt(var.clamp(min=1e-12))
                # pyro.sample("y", dist.Normal(mean, sd).to_event(1))
                pyro.sample("y", dist.StudentT(df=self.df_real, loc=mean, scale=sd).to_event(1))
                pyro.deterministic("mean", mean)                # read mean
        return model
    
    # only synthetic_data
    def make_realmodel(self):
        
        def model(xi):
            theta_log = pyro.sample("theta", self.theta_prior)
            with pyro.plate_stack("plates", xi.shape[:-1]):
                mean = self._forward_map_real(xi, theta_log)
                sd = torch.sqrt(0.01 * mean ** 2 + 0.1)
                pyro.sample("y", dist.Normal(mean, sd).to_event(1))
                pyro.deterministic("mean_real", mean)
        return model

    def synthetic_data(self, design: torch.Tensor, true_theta_log: torch.Tensor) -> torch.Tensor:
        """

        """
        conditioned = pyro.condition(self.make_realmodel(), data={"theta": true_theta_log})
        predictive = Predictive(conditioned, return_sites=["y"],num_samples=1)
        with torch.no_grad():
            out = predictive(design)
            y = out["y"]  # [..., 1]
        return y.detach().clone()

    def predict_data(self, design: torch.Tensor, theta_log: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        """

        """
        conditioned = pyro.condition(self.make_model(), data={"theta": theta_log})
        predictive = Predictive(conditioned, return_sites=["y"],num_samples=1)
        with torch.no_grad():
            out = predictive(design)
            y = out["y"]
        return y.detach().clone()