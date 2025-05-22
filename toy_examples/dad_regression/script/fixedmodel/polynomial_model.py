import torch
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive

 # Standard deviation for noise

class PolynomialModel:
    def __init__(self, 
        degree=2, 
        prior_mean=None, 
        prior_sd=None,
        TRUE_BETA = torch.tensor([1.0, 2.0, -0.5]),
        BEST_BETA = torch.tensor([-1.6720 + 2.0000]),
        NOISE_STD = 0.1,
        ):
        super().__init__()

        self.degree = degree
        self.prior_mean = prior_mean or torch.zeros(degree + 1)
        self.prior_sd = prior_sd or torch.ones(degree + 1)
        self.TRUE_BETA = TRUE_BETA
        self.BEST_BETA = BEST_BETA
        self.NOISE_STD = NOISE_STD

    

    def reset(self):
        self.prior_mean = torch.zeros(self.degree + 1)
        self.prior_sd = torch.ones(self.degree + 1)

    def make_model(self):
        def model( x):
            """Model function for Bayesian inference."""
            with pyro.plate_stack("plates", x.shape[:-1]):
                beta = pyro.sample("beta", dist.Normal(self.prior_mean, self.prior_sd).to_event(1))
                f_x = sum(beta[i] * x**i for i in range(self.degree + 1))
                # f_x = sum(beta[i] * x**i for i in range(self.degree)) # mis settings
                y = pyro.sample("y", dist.Normal(f_x, self.NOISE_STD).to_event(1))
            return y
        return model
    
    def make_mis_model(self):
        def mis_model( x):
            """Model function for Bayesian inference."""
            with pyro.plate_stack("plates", x.shape[:-1]):
                beta = pyro.sample("beta", dist.Normal(self.prior_mean, self.prior_sd).to_event(1))
                f_x = sum(beta[i] * x**i for i in range(self.degree))
                y = pyro.sample("y", dist.Normal(f_x, self.NOISE_STD).to_event(1))
            return y
        return mis_model


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
        pyro.sample("y", dist.Normal(q_mean, 1).to_event(1))

    def guide(self, x):
    # def guide(self, *args, **kwargs):
        posterior_mean = pyro.param("posterior_mean", self.prior_mean.clone())
        posterior_sd = pyro.param("posterior_sd", self.prior_sd.clone(), constraint=positive)
        pyro.sample("beta", dist.Normal(posterior_mean, posterior_sd).to_event(1))

    def compute_rmse(self, posterior_mean, true_params):
        """Computes RMSE between posterior mean and true parameters."""
        
        return torch.sqrt(torch.mean((posterior_mean - true_params)**2)).item()

