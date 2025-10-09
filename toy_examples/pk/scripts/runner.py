# runner.py
import math
import re
import torch
from torch import nn
import sys
from boed_experiment import BOEDExperiment 
from pk_model import PKModel,PKModel_Mis  
import gc
from plotting import compute_mmd_dispersion

def compute_mmd_biased_batched(X, sigma=1.0):
    """
    MMD^2
    X: (B, N, *feature_dims) 或 (N,) / (N,D)
    Y: (M, *feature_dims)   或 (M,) / (M,D)
    return: (B,)  batch 的 MMD^2(biased)
    """
    # ----  (B,N,D) and (M,D)
    Y = (0.0 + (24.0 - 0.0) * torch.rand(2000, 1, 1))
    if X.dim() == 1:
        X = X.unsqueeze(0).unsqueeze(-1)  # (1,N,1)
    elif X.dim() == 2:
        X = X.unsqueeze(0)                # (1,N,D)
    B, N, *fd = X.shape
    D = math.prod(fd)
    X = X.reshape(B, N, D)                # (B,N,D)

    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)               # (M,1)
    M = Y.shape[0]
    Y = Y.reshape(M, D)                   # (M,D)

    # ---- RBF kernel 的 pairwise 距离并向量化
    def rbf_K(x, y):
        """
        x: (B,N,D), y: (M,D) -> K: (B,N,M)
        或 x: (B,N,D), y: (B,N,D) -> K: (B,N,N)（用于 Kxx）
        """
        if y.dim() == 2:  # (M,D) 情况
            # (B,N,1) + (1,1,M) - 2*(B,N,M)
            x2 = (x**2).sum(-1, keepdim=True)              # (B,N,1)
            y2 = (y**2).sum(-1).view(1, 1, y.shape[0])     # (1,1,M)
            xy = x @ y.t()                                  # (B,N,M)
            dist2 = x2 + y2 - 2.0 * xy                      # (B,N,M)
        else:  # (B,N,D) 情况（Kxx）
            # (B,N,1) + (B,1,N) - 2*(B,N,N)
            x2 = (x**2).sum(-1, keepdim=True)              # (B,N,1)
            dist2 = x2 + x2.transpose(1, 2) - 2.0 * (x @ x.transpose(1, 2))  # (B,N,N)
        return torch.exp(-dist2 / (2.0 * sigma * sigma))

    # ---
    K_xx = rbf_K(X, X)          # (B,N,N)  
    K_xy = rbf_K(X, Y)          # (B,N,M)
    # 
    Y_ = Y.unsqueeze(0)         # (1,M,D)
    K_yy = rbf_K(Y_.expand(1, -1, -1), Y)  # 
    K_yy = K_yy.squeeze(0)      # (M,M)

    # ---- Biased MMD^2
    m = float(N)
    n = float(M)
    term_xx = K_xx.sum(dim=(1, 2)) / (m * m)     # (B,)
    term_yy = K_yy.sum() / (n * n)               # 标量
    term_xy = 2.0 * K_xy.sum(dim=(1, 2)) / (m * n)  # (B,)
    mmd2 = term_xx + term_yy - term_xy           # (B,)
    return mmd2


def test_phase(model, test_x, true_theta_log, num_samples=100, misspecification_flag=False):
    #  synthetic_data
    true_y = model.synthetic_data(test_x, true_theta_log)

    # posterior sampling（log-space）
    posterior_samples = torch.distributions.MultivariateNormal(
        model.theta_loc, model.theta_covmat
    ).sample(sample_shape=(num_samples,))

    # predict
    pred = torch.stack([model.predict_data(test_x, th) for th in posterior_samples])  # [S,N,1]
    pred_mean = pred.mean(dim=0)
    ge = torch.mean((true_y - pred_mean) ** 2).item()
    print(f"Generalization Error (MSE): {ge}")
    return ge


def single_run(
    num_candidates=200,
    eig_method="nmc_eig",
    dir="./results/",
    num_experiments=30,
    alg_method="db_boed",
    true_theta=None,                 # log-space
    biased_sample_flag=False,
    run_id_tensors={},
    misspecification_flag=False,
    design_space=(0.0, 24.0, 0.1),
):
    if not misspecification_flag:
        print("well specification!!!!")
        model = PKModel()
    else:
        print("MIS specification!!!!")
        model = PKModel_Mis()
    # 

    experiment = BOEDExperiment(
        model,
        num_candidates=num_candidates,
        eig_method=eig_method,
        dir=dir,
        biased_sample_flag=biased_sample_flag,
        design_space=design_space,
    )

    xis = torch.tensor([])
    ys = torch.tensor([])
    model.reset()

    run_mean = [model.theta_loc]
    run_std = [model.theta_covmat]
    rmse_all = [model.compute_rmse(model, true_theta)]
    run_nll = [torch.tensor(100.0)]
    xis_all = [torch.tensor([0.0])]  
    ge_all = []
    mmd_all = [torch.tensor(100.0)]


    num_test_points = 200
    test_x = experiment.initialize_candidate_designs(num_candidates=num_test_points, seed=42)
    ge_all.append(test_phase(model, test_x, true_theta, misspecification_flag=misspecification_flag))

    ref = torch.rand(2000, 1) * (24.0 - 0.0) + 0.0  
    for round_idx in range(num_experiments):
        print(f"=== Round {round_idx} ===")

        if alg_method == "boed" or alg_method.startswith("boed_new"):
            best_x = experiment.optimize_design(xis, ys, model, alg_method)
        elif alg_method in ("random", "random_new"):
            grid = torch.arange(design_space[0], design_space[1] + 1e-8, design_space[2])
            best_x = grid[torch.randint(0, len(grid), (1,))]  # [1]
        elif alg_method.startswith("db_boed"):
            # tau_/_alpha_/_lambda_/_test_...
            def parse_db_args(method: str):
                args = dict(tau=None, lengthscale=0.8, lam=1e-2, smooth_kappa=1.0,
                            test=False, steps=500, alpha=1.0, **{"lambda": 0.0})
                m = re.search(r"tau_([0-9.]+)", method);  args["tau"] = float(m.group(1)) if m else None
                m = re.search(r"alpha_([0-9.]+)", method);args["alpha"] = float(m.group(1)) if m else 1.0
                m = re.search(r"lambda_([0-9.]+)", method);args["lambda"] = float(m.group(1)) if m else 0.0
                m = re.search(r"test_(True|False)", method);args["test"] = (m.group(1)=="True") if m else False
                return args

            db_args = parse_db_args(alg_method)

            if round_idx == 0:
                grid = torch.arange(design_space[0], design_space[1] + 1e-8, design_space[2])
                best_x = grid[torch.randint(0, len(grid), (1,))]
            else:
                best_x = experiment.optimize_design(xis, ys, model, alg_method="db_boed", db_args=db_args)

        else:
            assert("wrong alg_method!, need to be boed, random,db_boed")
    
        if best_x.ndim == 0:
            best_x = best_x.unsqueeze(0)


        
        y = model.synthetic_data(best_x, true_theta)
        print("select design!:",best_x,y)
        print("shape", best_x.shape,y.shape)
        xis = torch.cat([xis, best_x.unsqueeze(0)])          # [n,1]
        ys = torch.cat([ys, y])                # [n,1,1]  to_event(1)

        posterior_mean, posterior_sd = experiment.run_svi(xis, ys)
        model.theta_loc = posterior_mean
        model.theta_covmat = posterior_sd
        model.set_theta_prior()

        rmse = model.compute_rmse(model, true_theta)
        loss_nll = nn.GaussianNLLLoss()
        nll = loss_nll(true_theta, posterior_mean, posterior_sd.diag())

        ge_all.append(test_phase(model, test_x, true_theta, misspecification_flag=misspecification_flag))
        mmd_d = compute_mmd_dispersion(xis)
        print("mmd_d:",mmd_d)

        mmd_all.append(mmd_d)
        rmse_all.append(rmse)
        run_mean.append(posterior_mean.numpy())
        run_std.append(posterior_sd.numpy())
        run_nll.append(nll)
        xis_all.append(best_x)

    torch.cuda.empty_cache()

    # clear
    if round_idx != 0:
        del model, experiment
        del xis, ys
        del posterior_mean, posterior_sd
        gc.collect()

    return xis_all, run_mean, run_std, rmse_all, run_nll, ge_all,mmd_all