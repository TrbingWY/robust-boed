import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from torch import nn
from torch.distributions.constraints import positive

class SourceModel:
    def __init__(self, base_signal=0.1, max_signal=1e-4, theta_loc=None, theta_covmat=None, noise_scale=0.5, p=1, K=1):
        self.base_signal = base_signal
        self.max_signal = max_signal
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, p))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(p).repeat(K, 1, 1)
        self.theta_prior = dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(1)
        self.noise_scale = noise_scale
        self.n = 1
        self.p = p
        self.K = K

    def set_theta_prior(self):
        self.theta_prior = dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(1)

    def reset(self):
        self.theta_loc = torch.zeros((self.K, self.p))
        self.theta_covmat = torch.eye(self.p).repeat(self.K, 1, 1)
        self.set_theta_prior()

    def make_model(self):
        def model(x):
            with pyro.plate_stack("plates", x.shape[:-1]):
                theta = pyro.sample("theta", dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(1))
                mean = self.forward_map(x, theta)
                sd = self.noise_scale
                y = pyro.sample("y", dist.Normal(mean, sd).to_event(1))
            return y
        return model

    def forward_map(self, xi, theta):
        if xi.dim() == 2:
            xi = xi.unsqueeze(1).repeat(1, 2, 1)
        elif xi.dim() == 3:
            xi = xi.unsqueeze(-2).repeat(1, 1, 2, 1)
        elif xi.dim() == 4:
            xi = xi.unsqueeze(-2).repeat(1, 1, 1, 2, 1)
        sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y

    def marginal_guide(self, design, observation_labels, target_labels):
        q_mean = pyro.param("q_mean", torch.zeros(design.shape[-2:]))
        q_sd = pyro.param("q_sd", torch.ones(design.shape[-2:]), constraint=constraints.positive)
        pyro.sample("y", dist.Normal(loc=q_mean, scale=q_sd).to_event(1))

    def guide(self, xi):
        with pyro.plate_stack("plate", xi.shape[:-1]):
            posterior_mean = pyro.param("posterior_mean", self.theta_loc.clone())
            posterior_sd = pyro.param("posterior_sd", self.theta_covmat.clone(), constraint=positive)
            pyro.sample("theta", dist.MultivariateNormal(posterior_mean, posterior_sd).to_event(1))

    def synthetic_data(self, design, true_theta):
        cond_model = pyro.condition(self.make_model(), data={"theta": true_theta})
        y = cond_model(design)
        return y.detach().clone()

    def compute_rmse(self, model, true_theta):
        mean, cov = model.theta_loc, model.theta_covmat
        dis_rmse = torch.tensor([])
        for _ in range(10000):
            sample_theta = pyro.sample("sample_theta", dist.MultivariateNormal(mean, covariance_matrix=cov))
            rmse_i = torch.sqrt(torch.mean((sample_theta - true_theta) ** 2, dim=1))
            dis_rmse = torch.cat((dis_rmse, rmse_i.unsqueeze(0)), dim=0)
        dis_rmse = torch.mean(dis_rmse, dim=0)
        print("MCMC case RMSE:", dis_rmse)
        return dis_rmse


class SourceModel_Mis:
    """Location finding example"""

    def __init__(
        self,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        noise_scale=0.5,  # this is the scale of the noise term
        p=1,  # physical dimension
        K=1,  # number of sources
    ):
        super().__init__()
        self.base_signal = base_signal
        self.max_signal = max_signal
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, p))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(p).repeat(K,1,1)
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
        # Observations noise scale:
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(1.0)
        self.n = 1  # batch=1
        self.p = p  # dimension of theta (location finding example will be 1, 2 or 3).
        self.K = K  # number of sources
    
    def set_theta_prior(self):
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)

    def reset(self):
        self.theta_loc =  torch.zeros((self.K, self.p))
        self.theta_covmat = torch.eye(self.p).repeat(self.K,1,1)
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)

    def make_model(self):
        def model(x):
            """Model function for Bayesian inference."""
            with pyro.plate_stack("plates", x.shape[:-1]):
                theta = pyro.sample("theta", dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(1))
                mean = self.forward_map(x, theta)
                sd = self.noise_scale

                y = pyro.sample(f"y", dist.Normal(mean, sd).to_event(1))
            return y
        return model
    
    def make_realmodel(self):
        def model(x):
            """Model function for Bayesian inference."""
            with pyro.plate_stack("plates", x.shape[:-1]):
                theta = pyro.sample("theta", dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(1))
                mean = self.forward_realmap(x, theta)
                sd = self.noise_scale

                y = pyro.sample(f"y", dist.Normal(mean, sd).to_event(1))
            return y
        return model
    
    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        # print("original forward")
        # two norm squared
        if xi.dim() == 2:
            xi = xi.unsqueeze(1).repeat(1,2,1)
        elif xi.dim() == 3:
            xi = xi.unsqueeze(-2).repeat(1,1,2,1)
        elif xi.dim() == 4:
            xi = xi.unsqueeze(-2).repeat(1,1,1,2,1)
        sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y
    
    def forward_realmap(self, xi, theta):
        # two norm squared
        # print("new forward map")
        if xi.dim() == 2:
            xi = xi.unsqueeze(1).repeat(1,2,1)
        elif xi.dim() == 3:
            xi = xi.unsqueeze(-2).repeat(1,1,2,1)
        elif xi.dim() == 4:
            xi = xi.unsqueeze(-2).repeat(1,1,1,2,1)
        eta = 2
        gain_factor = 0.4
        sq_two_norm = (xi - theta).pow(eta).sum(axis=-1)
        sq_two_norm_inverse =gain_factor * (self.max_signal*4 + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal*4 + sq_two_norm_inverse.sum(-1, keepdim=True))

        return mean_y
    
    def marginal_guide(self, design, observation_labels, target_labels):

        q_mean = pyro.param("q_mean", torch.zeros(design.shape[-2:]))
        q_sd = pyro.param("q_sd", torch.ones(design.shape[-2:]), constraint=dist.constraints.positive)

        y = pyro.sample("y", dist.Normal(loc=q_mean, scale=q_sd).to_event(1))

    def guide(self,xi):
        with pyro.plate_stack("plate", xi.shape[:-1]):
            posterior_mean = pyro.param("posterior_mean", self.theta_loc.clone())
            posterior_sd = pyro.param("posterior_sd", self.theta_covmat.clone(), constraint=positive)

            pyro.sample("theta", dist.MultivariateNormal(posterior_mean, posterior_sd).to_event(1))

    def synthetic_data(self, design, true_theta):
        """
        Execute an experiment with given design.
        """
        # create model from sampled params
        cond_model = pyro.condition(self.make_realmodel(), data={"theta":true_theta})

        # infer experimental outcome given design and model
        y = cond_model(design)
        y = y.detach().clone()
        return y
    
    def predict_data(self, design, true_theta):
        """
        Execute an experiment with given design.
        """
        # create model from sampled params
        cond_model = pyro.condition(self.make_model(), data={"theta":true_theta})

        # infer experimental outcome given design and model
        y = cond_model(design)
        y = y.detach().clone()
        return y
    
    def compute_rmse(self, model, true_theta):
        mean,cov = model.theta_loc, model.theta_covmat
        dis_rmse = torch.tensor([])
        for _ in range(10000):
                sample_theta = pyro.sample("sample_theta", dist.MultivariateNormal(mean, covariance_matrix=cov))
                rmse_i = torch.sqrt(torch.mean((sample_theta - true_theta) ** 2, dim=1))
                dis_rmse = torch.cat((dis_rmse, rmse_i.unsqueeze(0)), dim=0)
        dis_rmse = torch.mean(dis_rmse, dim=0)
        
        # dis_rmse = torch.sqrt(torch.mean((model.theta_loc - true_theta) ** 2, dim=1))
        # dis_rmse = torch.mean(dis_rmse, dim=0)
        print("MCMC case RMSE:", dis_rmse)
        return dis_rmse

   
    
