import mlflow
import torch
import random
from torch import nn
from boed_experiment import BOEDExperiment
from sourcemodel import SourceModel,SourceModel_Mis
import gc
from oed.primitives import compute_design


def test_phase(model, test_x, true_theta, num_samples=100,misspecification_flag= False):
    true_y = model.synthetic_data(test_x, true_theta)
    posterior_samples = torch.distributions.MultivariateNormal(
        model.theta_loc, model.theta_covmat
    ).sample(sample_shape=(num_samples,))

    if not misspecification_flag:
        predicted_y_samples = torch.stack(
            [model.synthetic_data(test_x, theta_sample) for theta_sample in posterior_samples]
        )
    else:
        predicted_y_samples = torch.stack(
            [model.predict_data(test_x, theta_sample) for theta_sample in posterior_samples]
        )  # Shape: (num_samples, test_points, 1)

    predicted_y_mean = predicted_y_samples.mean(dim=0)
    generalization_error = torch.mean((true_y - predicted_y_mean) ** 2).item()
    print(f"Generalization Error (MSE): {generalization_error}")
    return generalization_error


def single_run(
    p=2, 
    K=2, 
    noise_scale=0.5, 
    num_candidates=200, 
    eig_method="marginal_eig", 
    dir="./results/", 
    num_experiments=30, 
    alg_method="boed", 
    true_theta=None, 
    biased_sample_flag=False, 
    misspecification_flag= False,
    model_location = "",
    ):
    if not misspecification_flag:
        model = SourceModel(p=p, K=K, noise_scale=noise_scale)
    else:
        model = SourceModel_Mis(p=p, K=K, noise_scale=noise_scale)
    experiment = BOEDExperiment(model, num_candidates=num_candidates, eig_method=eig_method, dir=dir, biased_sample_flag=biased_sample_flag)

    xis = torch.tensor([])
    ys = torch.tensor([])
    model.reset()

    run_mean = [model.theta_loc]
    run_std = [model.theta_covmat]
    rmse_all = [model.compute_rmse(model, true_theta)]
    run_nll = [torch.tensor(100.0)]
    xis_all = [torch.tensor([0.0, 0.0])]
    generalization_error_all = []

    num_test_points = 200
    test_x = experiment.initialize_candidate_designs(num_candidates=num_test_points, seed=42)
    generalization_error_all.append(test_phase(model, test_x, true_theta,misspecification_flag=misspecification_flag))

    for round_idx in range(num_experiments):
        print(f"=== Round {round_idx} ===")

        if alg_method == "boed" or alg_method == "boed_new":
            best_x = experiment.optimize_design(xis, ys, model,alg_method)
            print(f"best_x:{best_x}")
        elif alg_method == "random":
            if not biased_sample_flag: 
                grid = torch.linspace(-4, 4, steps=int(8/0.1) + 1)  # [-4, ..., 4], step=0.1
                best_x = grid[torch.randint(0, len(grid), (model.p,))]
              
                # best_x = (torch.round(torch.rand((1, model.p)) * 8 / 0.1) * 0.1 - 4)[0]
            else:
                best_x = (torch.round(torch.rand((1, model.p)) * 3 / 0.1) * 0.1 - 4)[0]
        elif alg_method == "dad" or alg_method == "dad_new":
            ho_model = mlflow.pytorch.load_model(model_location)
            with torch.no_grad():
                best_x = compute_design(f"xi{round_idx}", ho_model.design_net.lazy(*zip(xis, ys)))

        if K == 1 and p == 1:
            best_x = best_x.unsqueeze(0)

        y = model.synthetic_data(best_x, true_theta)
        
        xis = torch.cat([xis, best_x.unsqueeze(0)])
        ys = torch.cat([ys, y.unsqueeze(-1)])

        posterior_mean, posterior_sd = experiment.run_svi(xis, ys)
        # posterior_mean, posterior_sd = experiment.run_mcmc(xis, ys)

        model.theta_loc = posterior_mean
        model.theta_covmat = posterior_sd
        model.set_theta_prior()

        rmse = model.compute_rmse(model, true_theta)
        loss_nll = nn.GaussianNLLLoss()
        nll = loss_nll(true_theta, posterior_mean,
                      torch.stack((posterior_sd[0].diag(), posterior_sd[1].diag()), dim=0))

        generalization_error_all.append(test_phase(model, test_x, true_theta,misspecification_flag=misspecification_flag))

        rmse_all.append(rmse)
        run_mean.append(posterior_mean.numpy())
        run_std.append(posterior_sd.numpy())
        run_nll.append(nll)
        xis_all.append(best_x)

    torch.cuda.empty_cache()

        # delte
    if round_idx != 0:
        del model, experiment
        del xis, ys
        del posterior_mean, posterior_sd
        gc.collect()  
    return xis_all, run_mean, run_std, rmse_all, run_nll, generalization_error_all
