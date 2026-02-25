import torch
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive

 # Standard deviation for noise

class PolynomialModel:
    def __init__(self, 
        degree=2, 
        prior_mean=None, 
        prior_scale_tril=None,
        TRUE_BETA = torch.tensor([1.0, 2.0, -0.5]),
        BEST_BETA = torch.tensor([-1.6720 + 2.0000]),
        NOISE_STD = 0.1,
        ):
        super().__init__()

        self.degree = degree
        d = degree + 1
        self.prior_mean = prior_mean if prior_mean is not None else torch.zeros(d)
      
        self.prior_scale_tril = prior_scale_tril if prior_scale_tril is not None else torch.eye(d)
        self.TRUE_BETA = TRUE_BETA
        self.BEST_BETA = BEST_BETA
        self.NOISE_STD = NOISE_STD

    

    def reset(self):
        self.prior_mean = torch.zeros(self.degree + 1)
        self.prior_scale_tril = torch.eye(self.degree + 1)

    # def make_model(self):
    #     def model( x):
    #         """Model function for Bayesian inference."""
    #         # with pyro.plate_stack("plates", x.shape[:-1]):
    #         #     beta = pyro.sample("beta", dist.Normal(self.prior_mean, self.prior_sd).to_event(1))
    #         #     f_x = sum(beta[i] * x**i for i in range(self.degree + 1))
    #         #     # f_x = sum(beta[i] * x**i for i in range(self.degree)) # mis settings
    #         #     y = pyro.sample("y", dist.Normal(f_x, self.NOISE_STD).to_event(1))
    #         # return y
        
    #         with pyro.plate_stack("plates",  x.shape[:-2]):  # 自己选合适的 dim
    #                     beta = pyro.sample(
    #                         "beta",
    #                         dist.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale_tril)
    #                     )  
    #                     powers = torch.cat([x**i for i in range(self.degree+1)], dim=-1)  # (..., F)
    #                     beta = beta.unsqueeze(-2)
                      
    #                     f_x = (powers * beta).sum(-1, keepdim=True)
               
    #         with pyro.plate_stack("plates_y",  x.shape[:-1]):
    #             y = pyro.sample("y", dist.Normal(f_x, self.NOISE_STD).to_event(1))
    #         return y
    #     return model
    


    def make_model(self):
        def model(x):
                with pyro.plate_stack("plates",  x.shape[:-2]):  # 自己选合适的 dim
                    beta = pyro.sample(
                        "beta",
                        dist.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale_tril)
                    )  # shape: (n_parallel, F)
                    
                    powers = torch.cat([x**i for i in range(self.degree+1)], dim=-1)  # (..., F)
                    beta = beta.unsqueeze(-2)
                    f_x = (powers * beta).sum(-1, keepdim=True)
                    # print("f_x.shape",f_x.shape)
                with pyro.plate_stack("plates_y",  x.shape[:-1]):
                    y = pyro.sample("y", dist.Normal(f_x, self.NOISE_STD).to_event(1))
                return y
        return model


    def synthetic_data(self, x):
        """Generates synthetic data based on the true coefficients."""
        noise = torch.randn_like(x) * self.NOISE_STD
        y = sum(self.TRUE_BETA[i] * x**i for i in range(len(self.TRUE_BETA))) + noise
        return y

 
    def synthetic_data_no_noise(self, x):
        """Generates synthetic data based on the true coefficients."""
        y = sum(self.TRUE_BETA[i] * x**i for i in range(len(self.TRUE_BETA)))
        return y

    def marginal_guide(self, design, observation_labels, target_labels):
        q_mean = pyro.param("q_mean", torch.zeros(design.shape[-2:]))
        pyro.sample("y", dist.Normal(q_mean, self.NOISE_STD).to_event(1))

    def guide(self, x):
        d = self.degree + 1
        loc = pyro.param("posterior_mean", self.prior_mean.clone())
        # 可学习的下三角矩阵，受 lower_cholesky 约束
        scale_tril = pyro.param(
            "posterior_scale_tril",
            self.prior_scale_tril.clone(),
            constraint=constraints.lower_cholesky
        )
        pyro.sample("beta", dist.MultivariateNormal(loc, scale_tril=scale_tril))

    def compute_rmse(self, posterior_mean, true_params):
        """Computes RMSE between posterior mean and true parameters."""
        
        return torch.sqrt(torch.mean((posterior_mean - true_params)**2)).item()

